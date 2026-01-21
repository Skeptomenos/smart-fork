# Smart Fork: Semantic Session Discovery for OpenCode

## The Problem

Every OpenCode session accumulates valuable context: understanding of codebases, solved problems, architectural decisions, debugging journeys, and domain knowledge. When a session ends, this context becomes trapped in transcript files—searchable only by filename or basic text grep.

**The cost of lost context:**

1. **Re-explanation overhead** - Starting fresh sessions means re-teaching Claude about your codebase, patterns, and preferences
2. **Repeated mistakes** - Solutions discovered in past sessions are forgotten, leading to repeated debugging cycles
3. **Missed connections** - Related work across different projects never cross-pollinates
4. **Cognitive load** - You must remember which session had the relevant context, or manually search through hundreds of transcripts

### The Status Quo

```
You: "I want to add webhook support to this API"
Claude: "I'd be happy to help! Can you tell me about your API structure?"
You: *sighs* "We literally did this exact pattern last month in another project..."
```

The `--resume` flag exists, but requires you to know *which* session to resume. With hundreds or thousands of sessions across dozens of projects, this becomes impossible.

## The Solution: Smart Fork

**Semantic session discovery powered by RAG.**

Instead of remembering session IDs or grepping transcripts, describe what you want to do in natural language. Smart Fork finds the sessions with the most relevant context and gives you a ready-to-paste fork command.

### How It Works

```
You: /detect-fork
Claude: "What are you trying to accomplish?"
You: "Add Stripe webhook handling with signature verification"

Claude: "Found 5 relevant sessions:

1. [94% match] gws-mcp-advanced - 3 days ago
   Implemented webhook receiver with HMAC verification for Google Push notifications
   
2. [87% match] account-management - 2 weeks ago  
   Built Stripe integration including checkout webhooks
   
3. [82% match] its-fusion-kitchen - 1 month ago
   Event-driven architecture with webhook dispatching

Fork from session 1?
→ opencode --resume abc123def"
```

### Two Scopes, One System

| Scope | Use Case | Query Filter |
|-------|----------|--------------|
| **Global** | Cross-pollinate learnings across all projects | No filter - search everything |
| **Repo** | Deep work within a single codebase | Filter by `repo_path` |

**Global scope** discovers patterns: "How did I handle rate limiting in that other project?"

**Repo scope** maintains continuity: "Continue the refactoring I started yesterday"

## Why This Matters

### Before Smart Fork
- Average 3-5 messages re-establishing context per session
- Repeated implementation of similar patterns
- Knowledge siloed within individual sessions
- Manual searching through transcript files

### After Smart Fork
- One command surfaces the most relevant prior context
- Patterns learned once, applied everywhere
- Sessions become a searchable knowledge base
- Fork from rich context, not blank slate

## The Compound Effect

Every session makes the system smarter. The 500th session benefits from the context of the previous 499. Your historical sessions become an ever-growing asset rather than forgotten artifacts.

```
Sessions completed: 1,247
Unique repos: 34
Total chunks indexed: 89,432
Average context reuse: 2.3 forks per new feature
```

This is not just about efficiency—it's about building a personal AI that actually learns from your work over time.

---

*"Don't let that valuable context go to waste."*
