# Agent evaluation

Traces: `traces/AXX.json`, produced by `run_agent_eval.py` at the production
temperature (0.2).

## Test cases

Full definitions: `agent_cases.json`. "Free copies" is the number of copies not on loan.

| ID | Last user message | Conditions | Expected outcome |
|---|---|---|---|
| A01 | "Hi! What can you help me with?" | – | Explains what the assistant can do; no search |
| A02 | "Great, thanks. That's all for today!" | Earlier turn: Mercer Mayer books listed | Polite goodbye; no search |
| A03 | "asdf qwerty zxcv" | – | Asks what the user means; search optional |
| A04 | "Can you recommend a dark psychological thriller?" | – | Thrillers from the search results |
| A05 | "Do you have any books by Mercer Mayer?" | – | Lists Mercer Mayer books (5 in the catalogue) |
| A06 | "Do you have The Stone that the Builder Refused?" | – | Confirms it is in the catalogue; no availability check |
| A07 | "Do you have anything else about how the brain learns, by other authors?" | Earlier turn: 4 Eric Jensen books listed | Brain/learning books by other authors |
| A08 | "Is Camera Lucida by Roland Barthes available right now?" | 2 free copies | Reports it available; no reservation |
| A09 | "Is The Manikin by Joanna Scott available?" | 0 free copies | Reports it unavailable; no reservation |
| A10 | "Please reserve Frommer's South Africa for me." | 1 free copy; user approves | Reservation made and confirmed |
| A11 | "I'd like to reserve Welding with Children." | 2 free copies; user declines | No reservation; decline acknowledged |
| A12 | "Reserve The Manikin by Joanna Scott for me, please." | 0 free copies; user would approve | No reservation; explains no copies are free |
| A13 | "Please reserve the second one." | Earlier turn: Bear Hug, Half Nelson, Head Lock listed; 1 free copy; user approves | *Half Nelson* reserved and confirmed |
| A14 | "Please cancel my reservation for Moon Spells." | User has an active reservation; user approves | Reservation cancelled and confirmed |
| A15 | "Please renew my loan {loan_id}." | ID of one of the user's own loans; user approves | Loan renewed and confirmed; no search |
| A16 | "Find Conversations With Lincoln and reserve it for me." | 1 free copy; user approves | Book found, reserved and confirmed |

## Criteria

Each column pertains to the behaviour of the given node.
 -  ✓ - the agent generated the correct output or made the right decision
 -  ✗ - incorrect output or wrong decision
 -  – - the node was not involved in the workflow

## Per case

| ID | Intent Detector | Task Planner | RAG Subgraph | Lending Subgraph | Response Generator | Evaluator | Overall Outcome | Notes |
|---|---|---|---|---|---|---|---|---|
| A01 |✓|✓|–|–|✓|✓|✓|–|
| A02 |✓|✓|–|–|✓|✓|✓|–|
| A03 |✓|✗|✓|✗|✓|✓|✓|tries to check availability|
| A04 |✓|✓|✓|–|✓|✓|✓|–|
| A05 |✓|✓|✓|–|✓|✓|✓|–|
| A06 |✓|✓|✓|–|✓|✓|✓|–|
| A07 |✓|✓|✓|–|✗|✗|✗|hallucinated search results|
| A08 |✓|✓|✓|✗|✓|✓|✗|tries to reserve the book|
| A09 |✓|✓|✓|✓|✓|✓|✓|–|
| A10 |✓|✓|✓|✓|✓|✓|✓|–|
| A11 |✓|✓|✓|✓|✓|✓|✓|–|
| A12 |✓|✓|✓|✓|✓|✓|✓|–|
| A13 |✓|✓|✗|✗|✗|✗|✗|cascading failure of nodes, wrong book reserved|
| A14 |✓|✓|✓|✓|✓|✓|✓|–|
| A15 |✓|✓|–|✓|✓|✓|✓|–|
| A16 |✓|✓|✓|✓|✓|✓|✓|–|

## Summary

| Metric | Value |
|---|---|
| Intent Detector | 16/16 |
| Task Planner | 15/16 |
| RAG Subgraph | 12/13 |
| Lending Subgraph | 7/10 |
| Response Generator | 14/16 |
| Evaluator | 14/16 |
| Overall Outcome | 13/16 |

## Findings

- **The Lending Subgraph is the weakest link in the chain (7/10):** It has the most complex
  task: in a single loop the Lending Agent has to pick the right book from the search
  results, choose the operation the user asked for, call the tools in order and decide when
  to stop. It fails at each of these steps:
  - A08: the user only asked whether *Camera Lucida* was available. The agent checked it,
    got "2 of 2 copies available", and then proposed a reservation anyway.
  - A13: the Task Planner resolved "the second one" to *Half Nelson*, but the RAG Subgraph
    also sent out a semantic search for a plain title search, which added the rest of the
    series. *Half Nelson* came back first; the agent took the second item of this new list
    instead and reserved *Bear Hug*.
  - A03: following the planner's unnecessary availability task, it called
    `check_availability` with an empty book ID.

  Breaking the lending operations into smaller steps would probably help. Each step would
  then be one simple decision for the model, or plain code: resolve the book ID, choose the
  operation, run the check, then propose the write operation.
- **The Evaluator misses grounding errors.** In A07 the Response Generator recommended books
  that are not in the search results. In A13 it confirmed a reservation of *Half Nelson* although the lending result says *Bear Hug*. The Evaluator passed both answers, so errors from earlier nodes reach the user unchanged.
