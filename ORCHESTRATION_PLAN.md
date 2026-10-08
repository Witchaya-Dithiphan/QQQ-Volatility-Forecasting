# M3–M10 Dynamic AI Orchestration Pipeline (No Muse)
## Hermes conductor, 3 AI models + intelligent routing by task complexity

---

## 1. Available AI Models & Characteristics

| Model | CLI | Strength | Cost | Speed | Context | Best For |
|---|---|---|---|---|---|---|
| **Gemini 2.0 Flash** | `gemini run --model gemini-2.0-flash` | Fast, cheap, good reasoning | ~$0.10/M tokens | ⚡ Fast | 1M tokens | Triage, monitoring, lightweight decisions |
| **Claude Sonnet 4.6** | `claude -p --model sonnet` | Complex logic, architecture, review | $3–15/M tokens | ⏱️ Medium | 200K context | Deep code review, complex algorithms, debugging |
| **Codex (ChatGPT 4-turbo)** | `codex --model gpt-4-turbo` | Code generation, TDD | Limited quota (~100K/week) | ⏱️ Medium | 128K context | TDD implementation, quick code fixes |

**Key constraint:** No free bulk generation → must optimize Codex quota + use Gemini for mechanical tasks

---

## 2. Task Complexity Classification & Routing

### Auto-classifier: Gemini triage → pick best model

```python
COMPLEXITY_TIERS = {
    "tier_1_trivial": {
        "complexity": 1,
        "agent": "gemini",
        "examples": [
            "Generate test skeleton for regression",
            "Adapt existing code (copy returns)",
            "Format documentation",
            "Update config value"
        ],
        "token_budget": 500,
        "cost": "$0.01",
        "rules": [
            "No new algorithm logic",
            "Pure copy-adapt or template gen",
            "No edge cases or special handling"
        ]
    },
    
    "tier_2_mechanical": {
        "complexity": 2,
        "agent": "gemini",
        "examples": [
            "Simple Linear → Polynomial (adapt degree)",
            "Generate 10 similar test cases (parametrized)",
            "Copy metric function with renamed variables",
            "Write boilerplate CLI wrapper"
        ],
        "token_budget": 1000,
        "cost": "$0.05",
        "rules": [
            "Clear algorithm pattern already exists",
            "Mechanical transformation only",
            "Can verify with simple pytest"
        ]
    },
    
    "tier_3_standard": {
        "complexity": 3,
        "agent": "codex",
        "examples": [
            "Multiple Linear Regression scratch (TDD)",
            "k-NN from scratch with distance metric",
            "Decision Tree from scratch",
            "Logistic Regression (gradient descent)"
        ],
        "token_budget": 3000,
        "cost": "$0.50–1.00",
        "rules": [
            "Clear algorithm from textbook",
            "Straightforward edge cases",
            "Existing reference implementations to check against",
            "TDD approach works well"
        ]
    },
    
    "tier_4_complex": {
        "complexity": 4,
        "agent": "codex_with_sonnet_arch",
        "examples": [
            "SVM from scratch (kernel trick, SMO)",
            "Stacking ensemble (holdout + meta-learner)",
            "MLP with backprop (weight initialization, convergence)",
            "Gradient Boosting (residuals, learning rate scheduling)"
        ],
        "token_budget": 5000,
        "cost": "$1.00–2.00",
        "rules": [
            "Non-trivial algorithm with subtleties",
            "Multiple ways to implement (Sonnet picks best approach)",
            "Edge cases: convergence, numerical stability",
            "Reference comparison essential"
        ]
    },
    
    "tier_5_bleeding_edge": {
        "complexity": 5,
        "agent": "sonnet_lead",
        "examples": [
            "XGBoost faithful implementation (second-order loss, regularization)",
            "Agglomerative clustering (linkage options, dendrogram)",
            "PCA with deterministic sign canonicalization",
            "Custom loss function with numerical stability"
        ],
        "token_budget": 8000,
        "cost": "$3.00–5.00",
        "rules": [
            "High-risk algorithm (leakage, numerical issues)",
            "Careful design needed (Sonnet architecture review first)",
            "Multiple implementation pitfalls",
            "Exhaustive testing + comparison required"
        ]
    }
}
```

---

## 3. Model Selection Strategy by Task

### Pattern A: Tier 1–2 (Trivial/Mechanical) → Gemini Only

```
Task: "Generate 5 regression test templates"
  ↓
[Gemini triage] "Tier 1 mechanical"
  ↓
[Gemini implement] 
  - Parametrized pytest skeletons
  - No algorithm logic needed
  - 5–10 min, ~300 tokens, $0.02
  ↓
[Hermes verify] pytest --co (check structure only)
  ↓
✅ Done
```

**Cost: $0.02 | Time: 15 min | Token budget impact: negligible**

---

### Pattern B: Tier 3 (Standard) → Codex (with Gemini monitoring)

```
Task: "Multiple Linear Regression scratch + tests"
  ↓
[Gemini triage] "Tier 3 standard, route to Codex TDD"
  ↓
[Codex TDD loop]
  - RED: write failing test
  - FIX: implement normal equation
  - GREEN: test passes
  - LOAD: save/reload test
  - Compare: vs sklearn reference
  ↓
[Gemini QA] "Does it follow contract? Any edge cases?"
  ↓
[Hermes verify] pytest full suite + artifact check
  ↓
✅ Done

Cost: ~$0.80 | Time: 2–3 hours | Token budget: ~1500 (Codex) + 300 (Gemini)
```

---

### Pattern C: Tier 4 (Complex) → Codex + Sonnet architecture first

```
Task: "SVM with linear/RBF kernels, scratch SMO solver"
  ↓
[Gemini triage] "Tier 4 complex, needs architecture"
  ↓
[Sonnet architecture phase]
  - Pseudocode for SMO
  - Kernel abstraction design
  - Numerical stability notes
  - Reference implementation sketch
  - 30 min, ~$1.00, 2K tokens
  ↓
[Codex implementation phase] (given Sonnet blueprint)
  - TDD with Sonnet skeleton
  - SMO loop + convergence handling
  - Kernel abstraction
  - 3–4 hours, ~$1.00, 2K tokens
  ↓
[Sonnet code review] (cached context)
  - Check for leakage/correctness
  - Compare vs sklearn decision boundary
  - 20 min, ~$0.50 (cache hit)
  ↓
[Hermes verify] pytest + visual sanity check
  ↓
✅ Done

Cost: ~$2.50 | Time: 4–5 hours | Token budget: 4.5K total
```

---

### Pattern D: Tier 5 (Bleeding edge) → Full orchestration

```
Task: "XGBoost faithful scratch (second-order, regularization)"
  ↓
[Gemini triage] "Tier 5 bleeding edge, CRITICAL"
  ↓
[Sonnet deep architecture]
  - Loss function design (hessian + regularization)
  - Tree growth strategy (gain calculation)
  - Shrinkage + column subsampling
  - Convergence proof sketch
  - Failure modes & how to test
  - 1 hour, ~$2.00, 5K tokens
  ↓
[Codex careful implementation] (step-by-step)
  - RED test: synthetic XOR problem
  - FIX: implement boosting loop
  - GREEN: test passes
  - Add: tree pruning by gain
  - Add: shrinkage (learning rate)
  - Compare: vs xgboost library on toy data
  - 6–8 hours, ~$3.00, 4K tokens
  ↓
[Sonnet exhaustive review]
  - Edge case: leaf with 1 sample
  - Edge case: negative gain (stop split)
  - Edge case: numerical precision (hessian near 0)
  - Numerical stability: clip gradients?
  - 1 hour, ~$2.00 (fresh cache due to complexity)
  ↓
[Hermes full verification]
  - pytest with 20+ edge cases
  - Convergence plot vs xgboost
  - Save/load/reload on real data
  - Artifact + manifest validation
  ↓
✅ Done

Cost: ~$7.00 | Time: 8–10 hours | Token budget: 9K total (high but justified)
```

---

## 4. M3–M10 Task Routing Matrix

| Task | Algorithm | Tier | Primary Agent | Support | Est. Time | Est. Cost |
|---|---|---|---|---|---|---|
| **M3** | Multiple Linear Reg | 3 | Codex | Gemini QA | 3h | $0.80 |
| **M4-01** | Simple Linear | 2 | Gemini | Hermes verify | 1.5h | $0.05 |
| **M4-02** | Multiple Linear | 3 | Codex | Gemini QA | 2h | $0.80 |
| **M4-03** | Polynomial | 2 | Gemini | Hermes verify | 1.5h | $0.05 |
| **M4-04** | Elastic Net | 3 | Codex | Sonnet review | 3h | $1.20 |
| **M5-01** | Logistic | 3 | Codex | Gemini QA | 2.5h | $0.80 |
| **M5-02** | Naive Bayes | 2 | Gemini | Hermes verify | 1.5h | $0.05 |
| **M5-03** | k-NN | 2 | Gemini | Hermes verify | 1.5h | $0.05 |
| **M5-04** | Perceptron | 3 | Codex | Gemini QA | 2h | $0.80 |
| **M5-05** | SLP | 3 | Codex | Gemini QA | 2h | $0.80 |
| **M5-06** | Decision Tree | 3 | Codex | Gemini QA | 2.5h | $0.80 |
| **M6-01** | Random Forest | 4 | Codex+Sonnet | Hermes verify | 4h | $2.00 |
| **M6-02** | Gradient Boosting | 4 | Codex+Sonnet | Hermes verify | 5h | $2.00 |
| **M6-03** | AdaBoost | 4 | Codex+Sonnet | Hermes verify | 4h | $2.00 |
| **M6-04** | SVM | 5 | Sonnet+Codex | Hermes verify | 5h | $3.00 |
| **M6-05** | MLP | 5 | Sonnet+Codex | Hermes verify | 6h | $3.50 |
| **M6-06** | Stacking | 5 | Sonnet+Codex | Hermes verify | 6h | $3.50 |
| **M6-07** | XGBoost | 5 | Sonnet+Codex | Hermes verify | 10h | $7.00 |
| **M7-01** | k-Means | 3 | Codex | Gemini QA | 2.5h | $0.80 |
| **M7-02** | Agglomerative | 4 | Codex+Sonnet | Hermes verify | 4h | $2.00 |
| **M7-03** | PCA | 4 | Codex+Sonnet | Hermes verify | 4h | $2.00 |
| **M8** | Report/Slides | 3–4 | Sonnet | Gemini review | 8h | $4.00 |
| **M9** | Non-Spike Reruns | — | Hermes loop | Auto-dispatch | 40h compute | $2.00 |
| **M10** | Paired Stats | 4 | Sonnet | Gemini verify | 6h | $3.00 |

**Total Effort:** ~140 hours (wall time ~50–70 hours with parallelism)  
**Total Cost:** ~$45–55  
**Codex quota needed:** ~25K tokens/week (manageable within 100K limit)  
**Sonnet budget:** ~$25–30 (high but essential for tier 5)

---

## 5. Hermes Orchestrator Loop (Pseudo-Python)

```python
#!/usr/bin/env python3
"""
m3_m10_conductor.py
Runs every 4 hours; dispatches tasks based on triage complexity
"""

import subprocess
import json
from pathlib import Path
from datetime import datetime
from hermes_tools import kanban_list, kanban_create, kanban_move, kanban_block

GEMINI_BUDGET_DAILY = 10000  # tokens
CODEX_BUDGET_DAILY = 5000    # tokens (split from 100K/week ~14K/day)
SONNET_BUDGET_DAILY = 3000   # tokens

def triage_task(task_brief: str) -> dict:
    """Use Gemini to classify task complexity and recommend agent."""
    prompt = f"""
Task: {task_brief}

Classify complexity (1-5) and recommend routing:
1 = trivial (copy/template)
2 = mechanical (adapt existing)
3 = standard (clear algorithm, TDD works)
4 = complex (needs architecture help)
5 = bleeding edge (risky, needs Sonnet lead)

Return JSON:
{{
  "tier": 1-5,
  "recommended_agent": "gemini|codex|sonnet_arch|sonnet_lead",
  "rationale": "...",
  "estimated_tokens": 500-8000,
  "estimated_hours": 0.5-10
}}
"""
    result = subprocess.run(
        ['gemini', 'run', '--model', 'gemini-2.0-flash', prompt],
        capture_output=True, text=True, timeout=60
    )
    return json.loads(result.stdout)

def dispatch_gemini(task_id: str, brief: str, tier: int):
    """Dispatch tier 1-2 or QA check to Gemini."""
    cmd = ['gemini', 'run', '--model', 'gemini-2.0-flash', '--auto', brief]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    kanban_move(task_id, status='running', assignee='gemini')
    return {'task_id': task_id, 'process': proc, 'agent': 'gemini'}

def dispatch_codex(task_id: str, brief: str, is_tdd: bool = True):
    """Dispatch tier 3-4 to Codex with TDD or code gen."""
    if is_tdd:
        cmd = ['codex', '--sandbox', 'workspace-write', '--ask-for-approval', 'never',
               '-c', 'model_reasoning_effort=high', 'exec', brief]
    else:
        cmd = ['codex', '--model', 'gpt-4-turbo', '--effort', 'high', brief]
    
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    kanban_move(task_id, status='running', assignee='codex')
    return {'task_id': task_id, 'process': proc, 'agent': 'codex'}

def dispatch_sonnet_arch(task_id: str, brief: str):
    """Dispatch architecture/design phase to Sonnet."""
    cmd = ['claude', '-p', '--model', 'sonnet', '--effort', 'high',
           '--permission-mode', 'plan', '--max-turns', '3', '--max-budget-usd', '3.00', brief]
    
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    kanban_move(task_id, status='running', assignee='sonnet_arch')
    return {'task_id': task_id, 'process': proc, 'agent': 'sonnet_arch'}

def dispatch_sonnet_review(task_id: str, code_diff: str):
    """Dispatch code review to Sonnet (cached context)."""
    cmd = ['claude', '-p', '--model', 'sonnet', '--effort', 'medium',
           '--permission-mode', 'plan', '--max-turns', '2', '--max-budget-usd', '1.00',
           f'Review this implementation:\n{code_diff}']
    
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    kanban_move(task_id, status='running', assignee='sonnet_review')
    return {'task_id': task_id, 'process': proc, 'agent': 'sonnet_review'}

def monitor_processes(processes: list) -> dict:
    """Check running processes; move to done/blocked based on exit code."""
    completed = {}
    for p_info in processes:
        proc = p_info['process']
        exit_code = proc.poll()
        
        if exit_code is not None:  # Finished
            stdout, stderr = proc.communicate()
            if exit_code == 0:
                kanban_move(p_info['task_id'], status='done')
                completed[p_info['task_id']] = 'done'
            else:
                kanban_block(p_info['task_id'], 
                           reason=f"{p_info['agent']} failed: {stderr.decode()[:200]}")
                completed[p_info['task_id']] = 'blocked'
    
    return completed

def main():
    """Orchestrator main loop."""
    print(f"\n{'='*60}")
    print(f"M3–M10 Orchestrator Loop | {datetime.now().isoformat()}")
    print(f"{'='*60}")
    
    # 1. Triage pending tasks
    pending = kanban_list(status='triage', limit=10)
    print(f"\n[1/4] Triaging {len(pending)} pending tasks...")
    
    for task in pending:
        triage_result = triage_task(task['body'])
        tier = triage_result['tier']
        agent = triage_result['recommended_agent']
        
        # Update kanban with triage result
        kanban_move(task['id'], status='ready', assignee=agent)
        print(f"  → {task['title']}: Tier {tier} → {agent}")
    
    # 2. Dispatch ready tasks (respecting budget)
    ready = kanban_list(status='ready', limit=5)
    print(f"\n[2/4] Dispatching {len(ready)} ready tasks...")
    
    processes = []
    for task in ready:
        assignee = task.get('assignee', 'unknown')
        
        if assignee in ('gemini', 'tier_1', 'tier_2'):
            p = dispatch_gemini(task['id'], task['body'], tier=int(assignee[-1]))
        elif assignee in ('codex', 'tier_3', 'tier_4'):
            p = dispatch_codex(task['id'], task['body'], is_tdd=True)
        elif assignee == 'sonnet_arch':
            p = dispatch_sonnet_arch(task['id'], task['body'])
        elif assignee == 'sonnet_review':
            # This is queued after Codex completes; find code diff in artifact
            code_diff = get_latest_artifact(task['id'])
            p = dispatch_sonnet_review(task['id'], code_diff)
        else:
            print(f"  ⚠️  Unknown assignee {assignee}, skip {task['title']}")
            continue
        
        processes.append(p)
        print(f"  → Dispatched {task['title']} to {p['agent']}")
    
    # 3. Monitor running tasks
    print(f"\n[3/4] Monitoring {len(processes)} running processes...")
    running = kanban_list(status='running')
    completed_now = monitor_processes(processes)
    
    for task_id, status in completed_now.items():
        print(f"  → {task_id}: {status}")
    
    # 4. Check for unblocked dependencies
    print(f"\n[4/4] Unblocking dependency-resolved tasks...")
    blocked = kanban_list(status='blocked')
    for task in blocked:
        if all(parent_done(p) for p in task.get('parents', [])):
            kanban_move(task['id'], status='ready')
            print(f"  → Unblocked {task['title']}")
    
    print(f"\n{'='*60}")
    print(f"Orchestrator cycle complete. Next run in 4 hours.")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()
```

---

## 6. Kanban Board Structure

```
Board: modeling-m3-m10 (GitHub-style columns)

Triage (new tasks land here)
  ├─ M3-02: MLR with-spike
  ├─ M4-01: Simple Linear
  └─ M6-07: XGBoost (⚠️ high risk)

Ready (triaged, awaiting dispatch)
  ├─ M4-01: Gemini (tier 2)
  ├─ M4-02: Codex (tier 3)
  └─ M6-04: Sonnet+Codex (tier 5)

Running (currently being worked on)
  ├─ [Gemini] M5-02: Naive Bayes → 10 min
  ├─ [Codex] M5-01: Logistic → 2h
  └─ [Sonnet] M6-07: XGBoost arch → 1h

Blocked (waiting on dependencies or error)
  ├─ M9-01: Non-Spike runs (blocked on M8 complete)
  └─ M6-06: Stacking (needs M5 complete)

Review (awaiting human approval or Sonnet review)
  ├─ M5-04: Perceptron (Sonnet review)
  └─ M6-07: XGBoost (Sonnet review)

Done (✅ verified & passing tests)
  ├─ M3-02: MLR (✅)
  ├─ M4-01: Simple Linear (✅)
  ├─ M4-02: Multiple Linear (✅)
  └─ M4-03: Polynomial (✅)
```

---

## 7. Token Budget Management

### Daily allocation (assume 100K Codex/week = ~14K/day):

```
Gemini 2.0 Flash:     2K/day (cheap, fast triage)
Codex (ChatGPT 4-t):  5K/day (core implementations)
Sonnet 4.6:           3K/day (complex + reviews)
Hermes (local pytest): unlimited

Weekly total: ~70K Codex + 14K Sonnet + 10K Gemini = ~$20–25
```

### Rationing rules:

1. **Tier 1–2 (Gemini only)**: Never block on Codex quota
2. **Tier 3 (Codex TDD)**: Use when fresh logic needed; skip if pattern clear
3. **Tier 4–5 (Sonnet+Codex)**: Reserve Sonnet for high-risk only; use Codex for implementation
4. **Fallback**: If Codex runs out, downgrade to Gemini code suggestions + Hermes manual implement

---

## 8. Implementation Checklist

- [ ] Commit M2 (✅ already done)
- [ ] Push to main (✅ already done)
- [ ] Create Hermes kanban board `modeling-m3-m10`
- [ ] Seed 40 tasks (M3–M10 breakdown)
- [ ] Implement `triage_task()` Gemini integration
- [ ] Implement dispatch functions (gemini, codex, sonnet)
- [ ] Test dispatch on M3 pilot (manual first)
- [ ] Deploy orchestrator cron (every 4 hours)
- [ ] Monitor daily + report progress
- [ ] Oct 12 11:59 PM: Final merge + submission

---

## 9. Expected Outcome

| Metric | Target |
|---|---|
| **All 19 algorithms implemented** | ✅ (tier routing ensures no skips) |
| **Both variants (with-spike + non-spike)** | ✅ (M3–M8 with-spike, M9 non-spike) |
| **Token spent** | ~$20–25 total |
| **Wall time** | ~60–80 hours continuous → ~3–5 days real-time |
| **Human intervention** | ~2 hours (initial setup + daily checks) |
| **Zero invented metrics** | ✅ (all tied to artifacts via kanban) |
| **Leakage-free** | ✅ (Sonnet reviews tier 5, Hermes verifies) |

**Ready?** ✅
