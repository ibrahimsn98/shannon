// Copyright (C) 2026 Keygraph, Inc.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License version 3
// as published by the Free Software Foundation.

/** Generic child-session delegation for the pi harness. */

import { createHash } from 'node:crypto';
import * as fs from 'node:fs';
import * as path from 'node:path';
import type { AssistantMessage, Model } from '@earendil-works/pi-ai';
import {
  createAgentSession,
  defineTool,
  getAgentDir,
  type ModelRuntime,
  type ResourceLoader,
  SessionManager,
  SettingsManager,
  type ToolDefinition,
} from '@earendil-works/pi-coding-agent';
import { Type } from 'typebox';
import { type LoggableAgentName, normalizeSemanticLabel } from '../../audit/safe-fields.js';
import { PI_RETRY_SETTINGS } from './retry-settings.js';
import { PI_THINKING_LEVEL } from './thinking-level.js';
import { TraceEmitter } from './trace-emitter.js';

export interface TaskToolContext {
  readonly cwd: string;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  readonly model: Model<any>;
  readonly modelRuntime: ModelRuntime;
  readonly createResourceLoader: () => Promise<ResourceLoader>;
  readonly parentAgentName: LoggableAgentName;
  readonly workflowLogPath?: string | undefined;
  readonly onDelegationStart?: ((child: string) => Promise<void>) | undefined;
  readonly cancellationSignal?: AbortSignal | undefined;
  readonly onUsage?: (usage: {
    readonly cost: number;
    readonly inputTokens: number;
    readonly outputTokens: number;
    readonly cacheReadTokens: number;
    readonly cacheWriteTokens: number;
  }) => void;
}

// Deliberately excludes `task` (no recursive delegation, so a child cannot spawn further children)
// and every collector/submit tool (structured output stays owned by the top-level agent session
// that the workflow reads back). A child session gets only plain file and shell access.
const CHILD_TOOLS = ['read', 'grep', 'find', 'ls', 'write', 'bash'];
const CHILD_FAILURE_TEXT = '[Sub-agent task failed before completion]';
const CHILD_CANCELLED_TEXT = '[Sub-agent task was cancelled]';

function textResult(text: string) {
  return { content: [{ type: 'text' as const, text }], details: undefined };
}

// --- shannon-scanner: sub-agent result cache --------------------------------
// Persist each successful sub-agent result under the run's writable overlay
// (.shannon/scratchpad), keyed by the exact task prompt. On a Temporal retry
// (e.g. after an LLM rate-limit or 5h quota failure) the parent re-issues the
// same task prompts; cached sub-agents return instantly and cost nothing, so the
// scan makes forward progress each quota window instead of restarting from zero.
const SUBAGENT_CACHE_SUBDIR = path.join('.shannon', 'scratchpad', 'subagent-cache');

function subagentCacheFile(cwd: string, prompt: string): string {
  const key = createHash('sha256').update(prompt).digest('hex');
  return path.join(cwd, SUBAGENT_CACHE_SUBDIR, `${key}.txt`);
}

function readSubagentCache(file: string): string | undefined {
  try {
    return fs.readFileSync(file, 'utf-8');
  } catch {
    return undefined;
  }
}

function writeSubagentCache(file: string, text: string): void {
  try {
    fs.mkdirSync(path.dirname(file), { recursive: true });
    fs.writeFileSync(file, text, 'utf-8');
  } catch {
    // Best-effort: a cache write must never break a scan.
  }
}

/**
 * Assigns each child a stable, safe display identity from its description. A duplicate of a
 * live sibling's name gets a monotonic start-order suffix (`route mapper #2`); a missing or
 * unsafe description becomes `subagent N`. State is shared across one parent's task calls,
 * and the assignment block runs synchronously so parallel calls never race on it.
 */
// Keep the base short enough that a `#N` suffix still fits the identity validator's length
// bound (48); a longer description falls back to `subagent N` rather than being dropped.
const MAX_CHILD_BASE_LENGTH = 40;

function createChildNamer(): (description: unknown) => string {
  const namedCounts = new Map<string, number>();
  let anonymousCount = 0;
  return (description) => {
    const base = normalizeSemanticLabel(description);
    if (base === undefined || base.length > MAX_CHILD_BASE_LENGTH) {
      anonymousCount += 1;
      return `subagent ${anonymousCount}`;
    }
    const nextOrdinal = (namedCounts.get(base) ?? 0) + 1;
    namedCounts.set(base, nextOrdinal);
    return nextOrdinal === 1 ? base : `${base} #${nextOrdinal}`;
  };
}

export function createTaskTool(config: TaskToolContext): ToolDefinition {
  const nameChild = createChildNamer();
  const logPath = config.workflowLogPath;

  return defineTool({
    name: 'task',
    label: 'Task',
    description:
      'Delegate a focused task to a sub-agent that runs independently with its own tools and returns ' +
      'the result. Use this to break complex work into smaller, parallelizable sub-tasks.',
    executionMode: 'parallel',
    promptSnippet: 'task - Delegate a focused task to a sub-agent with read, grep, find, ls, write, and bash.',
    promptGuidelines: [
      'Use the task tool to delegate focused work: code review, reconnaissance, automation scripting, validation.',
      'Pass all necessary context in the "prompt" parameter — the sub-agent cannot see your conversation history.',
      'The sub-agent can use read, grep, find, ls, write, and bash, but cannot call task or custom collector tools.',
      'You can launch multiple task tool calls in a single message to run sub-tasks in parallel.',
    ],
    parameters: Type.Object({
      prompt: Type.String({
        description: 'The task for the sub-agent to perform. Include all necessary context.',
      }),
      description: Type.Optional(Type.String({ description: 'A short (3-5 word) description of the task.' })),
    }),
    async execute(_toolCallId, params) {
      // Assign the identity synchronously, before any await, so concurrent siblings can't race.
      const child = nameChild(params.description);
      const emitter = logPath
        ? new TraceEmitter(logPath, { kind: 'child', parent: config.parentAgentName, child })
        : undefined;
      const startedAt = Date.now();

      // The parent's emitter first writes the raw task invocation, then this delegation
      // record. Awaiting it prevents the child emitter from overtaking its lineage start.
      await config.onDelegationStart?.(child);

      // shannon-scanner: return a cached result if this exact sub-agent task was
      // already completed in an earlier attempt of this scan (checkpoint resume).
      const cacheFile = subagentCacheFile(config.cwd, params.prompt);
      const cachedResult = readSubagentCache(cacheFile);
      if (cachedResult !== undefined) {
        console.error(`[SUBAGENT-CACHE] HIT ${child} ${path.basename(cacheFile)}`);
        emitter?.sessionComplete(Date.now() - startedAt, 0, 0);
        await emitter?.flush();
        return textResult(cachedResult);
      }
      console.error(`[SUBAGENT-CACHE] MISS ${child} ${path.basename(cacheFile)}`);

      const agentDir = getAgentDir();
      let subSession: Awaited<ReturnType<typeof createAgentSession>>['session'] | undefined;
      let resultText = '';
      let subCost = 0;
      let turns = 0;
      let operations = 0;
      let failed = false;
      let fatalFailure = false;

      const abortChildSession = (): void => {
        void subSession?.abort().catch(() => {
          // Dispose below still tears down the child session.
        });
      };
      const onCancellation = (): void => abortChildSession();

      try {
        const resourceLoader = await config.createResourceLoader();
        ({ session: subSession } = await createAgentSession({
          cwd: config.cwd,
          agentDir,
          resourceLoader,
          model: config.model,
          thinkingLevel: PI_THINKING_LEVEL,
          tools: CHILD_TOOLS,
          modelRuntime: config.modelRuntime,
          sessionManager: SessionManager.inMemory(config.cwd),
          settingsManager: SettingsManager.inMemory({
            retry: PI_RETRY_SETTINGS,
            compaction: { enabled: true },
          }),
        }));

        if (config.cancellationSignal?.aborted) {
          abortChildSession();
        } else {
          config.cancellationSignal?.addEventListener('abort', onCancellation, { once: true });
        }

        subSession.subscribe((event) => {
          if (event.type === 'tool_execution_start') {
            operations += 1;
            emitter?.toolStart(event.toolCallId, event.toolName, event.args);
            return;
          }
          if (event.type === 'tool_execution_end') {
            emitter?.toolEnd(event.toolCallId, event.isError);
            return;
          }
          if (event.type !== 'turn_end') return;
          turns += 1;
          const message = event.message as AssistantMessage | undefined;
          for (const block of message?.content ?? []) {
            if (block.type === 'text' && block.text) {
              resultText += (resultText ? '\n' : '') + block.text;
            }
          }
          if (message?.usage?.cost?.total != null) subCost += message.usage.cost.total;
        });

        try {
          await subSession.prompt(params.prompt);
        } catch {
          failed = true;
        }
        if (subSession.state.errorMessage !== undefined) failed = true;

        const subStats = subSession.getSessionStats();
        if (subStats.cost > subCost) subCost = subStats.cost;
        config.onUsage?.({
          cost: subCost,
          inputTokens: subStats.tokens.input,
          outputTokens: subStats.tokens.output,
          cacheReadTokens: subStats.tokens.cacheRead,
          cacheWriteTokens: subStats.tokens.cacheWrite,
        });
      } catch {
        fatalFailure = true;
      } finally {
        config.cancellationSignal?.removeEventListener('abort', onCancellation);
        subSession?.dispose();
      }

      const durationMs = Date.now() - startedAt;
      if (config.cancellationSignal?.aborted) {
        emitter?.sessionFailure('CANCELLED', durationMs);
        await emitter?.flush();
        return textResult(CHILD_CANCELLED_TEXT);
      }
      // `fatalFailure` means the child session itself never came up (createAgentSession threw), so
      // there is no session result to hand back, and this rethrows, which pi surfaces to the parent
      // as a failed tool call. `failed` means the session ran but ended in error; that gets a normal
      // text result instead, so the parent model sees the failure and can decide how to proceed.
      if (fatalFailure) {
        emitter?.sessionFailure('CHILD_TASK_FAILED', durationMs);
        await emitter?.flush();
        throw new Error(CHILD_FAILURE_TEXT);
      }
      if (failed) {
        emitter?.sessionFailure('CHILD_TASK_FAILED', durationMs);
        await emitter?.flush();
        return textResult(CHILD_FAILURE_TEXT);
      }

      // shannon-scanner: cache only clean successes (every failure/cancel path has
      // returned above) so a retry after a rate-limit / quota failure can skip this
      // sub-agent instead of re-running (re-paying). Failure is judged by session
      // state, never by scanning the text, whose security findings legitimately
      // mention "rate limit" / "429".
      if (resultText) {
        writeSubagentCache(cacheFile, resultText);
        console.error(`[SUBAGENT-CACHE] STORE ${child} ${path.basename(cacheFile)}`);
      }

      emitter?.sessionComplete(durationMs, turns, operations);
      await emitter?.flush();
      return textResult(resultText || '[Sub-agent produced no output]');
    },
  });
}
