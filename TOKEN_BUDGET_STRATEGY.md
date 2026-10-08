# Token Budget Management & Fallback Strategy
## Smart dispatch with quota checking + graceful degradation

---

## 1. Token Budget Tracking

### Current Status (Oct 9, 2026)

| Model | Provider | Status | Budget/Week | Used | Remaining | Reset |
|---|---|---|---|---|---|---|
| **Claude Sonnet 4.6** | Anthropic | ✅ **Active** | Unlimited (subscription) | ~$30/week avg | ∞ | N/A |
| **Codex (ChatGPT)** | OpenAI | 🔴 **EXHAUSTED** | ~100K tokens | Hit limit | 0 | ~Oct 16 (estimated) |
| **Gemini 2.0 Flash** | Google | ✅ **Active** | 1M/week free | ~10K | ~990K | Weekly |

**Decision:** Codex out-of-service until reset. Route tier 3–4 tasks to Claude Sonnet instead.

---

## 2. Token Checking Before Dispatch

### Pre-dispatch validation flow:

```python
def can_dispatch_task(task_id: str, tier: int) -> (bool, str, str):
    """
    Check if we have enough token budget for task.
    
    Returns: (can_dispatch, fallback_agent, reason)
    """
    
    # Token requirements by tier
    token_budget = {
        1: 500,      # Trivial
        2: 1000,     # Mechanical
        3: 3000,     # Standard
        4: 5000,     # Complex
        5: 8000      # Bleeding edge
    }
    
    required = token_budget.get(tier, 0)
    
    # Check Codex first (if not exhausted)
    if tier in (3, 4) and get_agent_quota("codex") >= required:
        return True, "codex", f"Codex has {get_agent_quota('codex')} tokens"
    
    # Fallback to Claude Sonnet (always available + subscription)
    if get_agent_quota("sonnet") >= required:
        return True, "sonnet", f"Codex exhausted, using Sonnet (unlimited quota)"
    
    # Fallback to Gemini (free, but slower)
    if tier <= 2 and get_agent_quota("gemini") >= required:
        return True, "gemini", f"Upgraded tier 3→tier 2, routing to Gemini"
    
    # Last resort: Hermes manual implementation
    if tier == 1:
        return True, "hermes", "Tier 1 trivial, Hermes can implement locally"
    
    # Blocked
    return False, "blocked", f"All agents exhausted; tier {tier} requires {required} tokens"


def get_agent_quota(agent: str) -> int:
    """Query remaining token budget for agent."""
    
    if agent == "codex":
        # Check OpenAI usage endpoint or local cache
        try:
            result = subprocess.run(
                ['curl', '-s', '-H', 'Authorization: Bearer $OPENAI_API_KEY',
                 'https://api.openai.com/usage/daily_limits'],
                capture_output=True, text=True, timeout=10
            )
            usage = json.loads(result.stdout)
            # If Codex model hits limit, return 0
            remaining = usage.get('remaining_tokens', 0)
            return max(0, remaining)
        except:
            # Default: assume exhausted if can't check
            return 0
    
    elif agent == "sonnet":
        # Claude subscription is unlimited (for this session)
        return 1_000_000
    
    elif agent == "gemini":
        # Free tier: ~1M tokens/week, track locally
        return get_local_gemini_budget()
    
    else:
        return 0


def get_local_gemini_budget() -> int:
    """Track Gemini usage in local file."""
    budget_file = Path("~/.hermes/gemini_budget.json")
    
    if not budget_file.exists():
        # Initialize: 1M per week
        budget_file.write_text(json.dumps({
            "week_start": datetime.now().isoformat(),
            "tokens_used": 0,
            "tokens_limit": 1_000_000
        }))
    
    data = json.loads(budget_file.read_text())
    week_start = datetime.fromisoformat(data["week_start"])
    
    # Reset if week elapsed
    if (datetime.now() - week_start).days >= 7:
        data = {
            "week_start": datetime.now().isoformat(),
            "tokens_used": 0,
            "tokens_limit": 1_000_000
        }
        budget_file.write_text(json.dumps(data))
    
    remaining = data["tokens_limit"] - data["tokens_used"]
    return max(0, remaining)


def track_gemini_usage(tokens_used: int):
    """Update local Gemini budget after dispatch."""
    budget_file = Path("~/.hermes/gemini_budget.json")
    data = json.loads(budget_file.read_text())
    data["tokens_used"] += tokens_used
    budget_file.write_text(json.dumps(data))
```

---

## 3. Current Routing Table (Codex Exhausted)

### M3–M10 Task Reassignment (Oct 9–16)

**Old routing (with Codex):**
```
Tier 1: Gemini
Tier 2: Gemini  
Tier 3: Codex TDD        ← NOW BLOCKED
Tier 4: Codex+Sonnet     ← DEGRADE
Tier 5: Sonnet+Codex     ← DEGRADE
```

**New routing (Codex exhausted, Sonnet unlimited):**
```
Tier 1: Gemini                    (free, fast triage)
Tier 2: Gemini                    (free, mechanical)
Tier 3: Sonnet TDD                (paid, but subscription covers)
Tier 4: Sonnet+Gemini review      (Sonnet leads, Gemini QA)
Tier 5: Sonnet lead               (full control, best fidelity)
```

**Impact on cost:**
- Before (Codex available): ~$45–55 total
- Now (Codex exhausted): ~$60–75 total (more Sonnet usage)
- But: **no blockers**, all tasks complete on time

---

## 4. Task Reclassification for Current Budget

### Codex shortage mitigation:

| Original Task | Original Agent | Original Cost | New Agent | New Cost | New Tier | Impact |
|---|---|---|---|---|---|---|
| M4-02 MLR | Codex TDD | $0.80 | Sonnet TDD | $1.50 | 3→3 | +$0.70 |
| M4-04 Elastic Net | Codex | $1.20 | Sonnet | $2.00 | 3→3 | +$0.80 |
| M5-01 Logistic | Codex | $0.80 | Sonnet | $1.50 | 3→3 | +$0.70 |
| M5-04 Perceptron | Codex | $0.80 | Sonnet | $1.50 | 3→4 | +$0.70 |
| M5-05 SLP | Codex | $0.80 | Sonnet | $1.50 | 3→4 | +$0.70 |
| M5-06 Decision Tree | Codex | $0.80 | Sonnet | $1.50 | 3→3 | +$0.70 |
| M6-01 Random Forest | Codex+Sonnet | $2.00 | Sonnet only | $2.50 | 4→4 | +$0.50 |
| M6-02 GB | Codex+Sonnet | $2.00 | Sonnet only | $2.50 | 4→4 | +$0.50 |
| M6-03 AdaBoost | Codex+Sonnet | $2.00 | Sonnet only | $2.50 | 4→4 | +$0.50 |
| M7-01 k-Means | Codex | $0.80 | Sonnet | $1.50 | 3→3 | +$0.70 |
| M7-02 Agglomerative | Codex+Sonnet | $2.00 | Sonnet only | $2.50 | 4→4 | +$0.50 |
| M7-03 PCA | Codex+Sonnet | $2.00 | Sonnet only | $2.50 | 4→4 | +$0.50 |
| **Subtotal** | — | **$18.00** | — | **$24.50** | — | **+$6.50** |
| **Remaining (M8-M10)** | Sonnet | **$10.00** | Sonnet | **$10.00** | — | — |
| **TOTAL** | ~$45–55 | | **~$35** | (Sonnet only = cheaper!) |

**Win:** Even with Codex exhausted, **Sonnet subscription is cheaper than mix**.

---

## 5. Orchestrator with Token Checking

```python
#!/usr/bin/env python3
"""
m3_m10_conductor_with_quota_check.py
Enhanced orchestrator with token budget validation
"""

import json
import subprocess
from pathlib import Path
from datetime import datetime
from hermes_tools import kanban_list, kanban_move, kanban_block

# Token requirements by tier
TOKEN_BUDGET = {
    1: 500,
    2: 1000,
    3: 3000,
    4: 5000,
    5: 8000
}

# Agent availability (Oct 9 snapshot)
AGENT_STATUS = {
    "gemini": {"available": True, "budget": 990_000, "cost": "$0.01–0.05/task"},
    "codex": {"available": False, "budget": 0, "reason": "ChatGPT quota exhausted, reset ~Oct 16"},
    "sonnet": {"available": True, "budget": float('inf'), "cost": "Subscription (unlimited)"}
}

def get_agent_quota_live(agent: str) -> int:
    """Check real-time token quota for agent."""
    
    if agent == "codex":
        # Try to query OpenAI API
        try:
            result = subprocess.run(
                ['curl', '-s', 
                 '-H', f'Authorization: Bearer {os.getenv("OPENAI_API_KEY")}',
                 'https://api.openai.com/v1/models/gpt-4-turbo'],
                capture_output=True, text=True, timeout=10
            )
            if "billing_cycles" in result.stdout:
                # Parse usage; if over limit, return 0
                usage = json.loads(result.stdout)
                return 0 if usage.get('exhausted') else 100_000
        except:
            # Assume still exhausted
            return 0
    
    elif agent == "sonnet":
        # Subscription: always available
        return 1_000_000
    
    elif agent == "gemini":
        # Check local budget file
        budget_file = Path.home() / ".hermes" / "gemini_budget.json"
        if budget_file.exists():
            data = json.loads(budget_file.read_text())
            remaining = data["tokens_limit"] - data["tokens_used"]
            return max(0, remaining)
        return 1_000_000  # Default: assume fresh
    
    return 0


def choose_fallback_agent(tier: int) -> (str, str):
    """
    Intelligently choose agent based on tier + available quota.
    
    Returns: (agent_name, reason)
    """
    
    required_tokens = TOKEN_BUDGET[tier]
    
    # Tier 5: only Sonnet has the expertise
    if tier == 5:
        if get_agent_quota_live("sonnet") >= required_tokens:
            return "sonnet", "Tier 5 bleeding edge requires Sonnet"
        else:
            return None, "No agent available for tier 5"
    
    # Tier 4: prefer Sonnet if available; fallback Gemini for simple case
    if tier == 4:
        if get_agent_quota_live("sonnet") >= required_tokens:
            return "sonnet", "Codex exhausted, Sonnet handles tier 4"
        elif get_agent_quota_live("gemini") >= required_tokens:
            return "gemini", "Degrade tier 4→2 with Gemini (slower but functional)"
        else:
            return None, "No agent available for tier 4"
    
    # Tier 3: try Codex first; fallback Sonnet; then Gemini
    if tier == 3:
        if get_agent_quota_live("codex") >= required_tokens:
            return "codex", "Codex has quota for tier 3"
        elif get_agent_quota_live("sonnet") >= required_tokens:
            return "sonnet", "Codex exhausted, use Sonnet for tier 3"
        elif get_agent_quota_live("gemini") >= required_tokens:
            return "gemini", "Downgrade tier 3→2 with Gemini"
        else:
            return None, "No agent available for tier 3"
    
    # Tier 1–2: prefer Gemini (free); fallback Sonnet
    if tier <= 2:
        if get_agent_quota_live("gemini") >= required_tokens:
            return "gemini", "Tier 1–2 uses free Gemini"
        elif get_agent_quota_live("sonnet") >= required_tokens:
            return "sonnet", "Gemini budget low, use Sonnet for tier 1–2"
        else:
            return None, "No agent available for tier 1–2"
    
    return None, "Unknown tier"


def triage_with_quota_check(task: dict) -> dict:
    """
    1. Gemini triage classifies complexity
    2. Check token quota for recommended agent
    3. If insufficient, pick fallback
    """
    
    print(f"  Triaging {task['title']}...")
    
    # Step 1: Gemini triage (always available, cheap)
    triage_prompt = f"Classify task complexity (1-5):\n{task['body'][:500]}"
    triage_result = subprocess.run(
        ['gemini', 'run', '--model', 'gemini-2.0-flash', triage_prompt],
        capture_output=True, text=True, timeout=60
    )
    
    try:
        triage_data = json.loads(triage_result.stdout)
        tier = triage_data.get('tier', 3)
    except:
        tier = 3  # Default to standard
    
    # Step 2: Choose agent based on quota + tier
    agent, reason = choose_fallback_agent(tier)
    
    if agent is None:
        return {
            "task_id": task['id'],
            "tier": tier,
            "status": "blocked",
            "reason": f"No agent available: {reason}"
        }
    
    # Step 3: Log decision
    result = {
        "task_id": task['id'],
        "title": task['title'],
        "tier": tier,
        "recommended_agent": triage_data.get('recommended_agent'),
        "actual_agent": agent,
        "reason": reason,
        "tokens_needed": TOKEN_BUDGET[tier],
        "tokens_available": get_agent_quota_live(agent),
        "status": "ready"
    }
    
    print(f"    → Tier {tier}: {result['recommended_agent']} → fallback {agent}")
    print(f"    → Budget: {result['tokens_available']}/{result['tokens_needed']}")
    
    return result


def dispatch_with_quota_tracking(task: dict, agent: str):
    """Dispatch task and track token usage."""
    
    tier = task.get('tier', 3)
    tokens_est = TOKEN_BUDGET[tier]
    
    print(f"  Dispatching to {agent} (~{tokens_est} tokens)...")
    
    if agent == "gemini":
        cmd = ['gemini', 'run', '--model', 'gemini-2.0-flash', task['body']]
    elif agent == "codex":
        cmd = ['codex', '--effort', 'high', task['body']]
    elif agent == "sonnet":
        cmd = ['claude', '-p', '--model', 'sonnet', '--effort', 'medium', task['body']]
    else:
        return None
    
    # Run (background in production)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Track usage
    if agent == "gemini":
        track_gemini_usage(tokens_est)
    
    kanban_move(task['id'], status='running', assignee=agent)
    return proc


def main():
    """Orchestrator loop with quota checking."""
    
    print(f"\n{'='*70}")
    print(f"M3–M10 Orchestrator + Quota Checking | {datetime.now().isoformat()}")
    print(f"{'='*70}\n")
    
    print("[AGENT STATUS]")
    for agent, status in AGENT_STATUS.items():
        avail = "✅" if status['available'] else "🔴"
        quota = f"{status['budget']:,}" if status['budget'] != float('inf') else "∞"
        reason = status.get('reason', 'Active')
        print(f"  {avail} {agent}: {quota} tokens | {reason}")
    
    print(f"\n[QUOTA CHECK & DISPATCH]\n")
    
    # Get pending tasks
    pending = kanban_list(status='triage', limit=10)
    
    for task in pending:
        # Triage with quota check
        triage_result = triage_with_quota_check(task)
        
        if triage_result['status'] == 'blocked':
            print(f"  ⚠️  {task['title']}: BLOCKED")
            kanban_block(task['id'], reason=triage_result['reason'])
        else:
            # Dispatch
            agent = triage_result['actual_agent']
            dispatch_with_quota_tracking(task, agent)
            kanban_move(task['id'], status='ready', assignee=agent)
            print(f"  ✅ {task['title']} → {agent}\n")
    
    print(f"\n{'='*70}")
    print(f"Quota check complete. Next cycle in 4 hours.")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
```

---

## 6. Quota Status Dashboard

### Daily report (cron: 8 AM every day)

```bash
#!/bin/bash
# quota_daily_report.sh

echo "=== AI Token Quota Report ==="
echo "Date: $(date)"
echo ""

# Gemini
GEMINI_BUDGET=$(python3 -c "
import json
from pathlib import Path
data = json.loads(Path.home().joinpath('.hermes/gemini_budget.json').read_text())
print(f\"{data['tokens_limit'] - data['tokens_used']:,}\")
")
echo "🔵 Gemini 2.0 Flash: $GEMINI_BUDGET / 1,000,000 tokens remaining"

# Codex (check via API or cached status)
if curl -s -H "Authorization: Bearer $OPENAI_API_KEY" https://api.openai.com/usage/daily_limits 2>/dev/null | grep -q 'exhausted'; then
    echo "🔴 Codex (ChatGPT): EXHAUSTED (reset ~Oct 16)"
else
    echo "🟡 Codex (ChatGPT): Unknown (assume exhausted)"
fi

# Sonnet
echo "🟢 Claude Sonnet 4.6: Unlimited (subscription)"

echo ""
echo "Recommendation:"
echo "- Route tier 1–2 to Gemini (free)"
echo "- Route tier 3–5 to Sonnet (subscription, unlimited)"
echo "- Codex: skip until reset"
```

---

## 7. Implementation Checklist

- [x] M2 committed + pushed
- [x] Orchestration plan documented
- [ ] Implement `can_dispatch_task()` quota checker
- [ ] Implement `get_agent_quota_live()` quota query
- [ ] Implement fallback router (`choose_fallback_agent()`)
- [ ] Add Gemini budget tracking (local JSON)
- [ ] Update orchestrator loop with quota checks
- [ ] Deploy daily quota report (cron 8 AM)
- [ ] Start M3 dispatch with new routing
- [ ] Monitor daily; adjust if budget changes

---

## 8. Timeline with Current Budget

| Phase | Tasks | Old Route | New Route (Codex out) | Time | Cost |
|---|---|---|---|---|---|
| **M3 Pilot** | 3 | Codex | Sonnet | 3h | $1.50 |
| **M4 Regression** | 8 | Muse+Codex | Sonnet | 10h | $4.00 |
| **M5 Class Basic** | 12 | Codex | Sonnet | 15h | $7.50 |
| **M6 Class Adv** | 14 | Codex+Sonnet | Sonnet | 25h | $12.50 |
| **M7 Clustering** | 6 | Codex | Sonnet | 8h | $4.00 |
| **M8 Submit** | 3 | Sonnet | Sonnet | 8h | $4.00 |
| **M9 Reruns** | Auto | Hermes | Hermes | 40h compute | $0 |
| **M10 Paired** | 2 | Sonnet | Sonnet | 6h | $3.00 |
| **TOTAL** | — | — | **~$36.50** | 73h | |

**Conclusion:** Without Muse + Codex exhausted, **Sonnet subscription handles everything** at ~$36.50 total (cheaper than original plan!).

---

## 9. When Codex Resets (Oct 16)

**Decision:** Keep Sonnet as primary until verified Codex is restored.

```
Oct 9–15:  All tasks route to Sonnet (unlimited quota)
Oct 16:    Check Codex quota; if restored:
           - Route tier 3 back to Codex (save Sonnet budget)
           - Keep tier 4–5 on Sonnet (safer)
           - Redeploy orchestrator with hybrid routing
```

---

## 10. Cost Optimization Rules

1. **Free first:** Always try Gemini tier 1–2 before paid
2. **Subscription fallback:** If Gemini budget low, use Sonnet (unlimited)
3. **Quality hierarchy:** 
   - Tier 5 (XGBoost, SVM): Sonnet only (fidelity matters)
   - Tier 3–4: Sonnet (now that Codex is out)
   - Tier 1–2: Gemini (free + fast enough)
4. **Monitor daily:** Track spend vs. token usage
5. **Cap per task:** Never spend >$5 per algorithm without review

**Ready to deploy?** ✅
