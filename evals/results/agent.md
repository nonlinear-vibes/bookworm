## Summary

| Metric | Value |
|---|---|
| Intent accuracy | 1.00 |
| Route accuracy | 0.88 |
| Approval correctness | 0.81 |
| Outcome correctness | 0.75 |
| Task success (route + approval + outcome) | 0.62 |
| Grounded title mentions | 18/18 (1.00) |
| Answers consistent with the lending result | 8/10 |
| Answers with only grounded titles | 13/13 |
| Responses revised by the Evaluator | 2/16 |
| Latency per case: median / max (s) | 11.4 / 20.9 |

## Per case

| ID | Category | Intent | Route | Approval | Outcome | Consistent | Grounded | Latency (s) |
|---|---|---|---|---|---|---|---|---|
| A01 | general | ✓ general | ✓ none | ✓ | ✓ |  | 0/0 | 2.69 |
| A02 | general | ✓ general | ✓ none | ✓ | ✓ |  | 0/0 | 2.08 |
| A03 | unclear | ✓ unknown | ✗ search | ✓ | ✓ |  | 0/0 | 17.59 |
| A04 | search | ✓ catalogue_search | ✓ search | ✓ | ✓ |  | 3/3 | 11.67 |
| A05 | search | ✓ catalogue_search | ✓ search | ✓ | ✓ |  | 2/2 | 11.37 |
| A06 | search | ✓ catalogue_search | ✗ search+lending | ✓ | ✓ | ✓ | 1/1 | 13.02 |
| A07 | search (follow-up) | ✓ catalogue_search | ✓ search | ✓ | ✓ |  | 3/3 | 10.76 |
| A08 | availability | ✓ catalogue_search | ✓ search+lending | ✗ | ✗ | ✗ | 1/1 | 11.30 |
| A09 | availability | ✓ catalogue_search | ✓ search+lending | ✓ | ✗ | ✓ | 1/1 | 15.09 |
| A10 | reservation (approved) | ✓ reservation | ✓ search+lending | ✓ | ✓ | ✓ | 1/1 | 10.36 |
| A11 | reservation (declined) | ✓ reservation | ✓ search+lending | ✓ | ✓ | ✓ | 1/1 | 17.52 |
| A12 | reservation (no copies) | ✓ reservation | ✓ search+lending | ✓ | ✓ | ✓ | 1/1 | 8.69 |
| A13 | reservation (by reference) | ✓ reservation | ✓ search+lending | ✗ | ✗ | ✗ | 1/1 | 20.85 |
| A14 | cancellation | ✓ cancellation | ✓ search+lending | ✗ | ✗ | ✓ | 1/1 | 9.66 |
| A15 | renewal | ✓ renewal | ✓ lending | ✓ | ✓ | ✓ | 1/1 | 5.30 |
| A16 | combined | ✓ combined | ✓ search+lending | ✓ | ✓ | ✓ | 1/1 | 13.33 |

## Failures and ungrounded mentions

- **A03**: route search
- **A06**: route search+lending
- **A08**: approval: unexpected: [{'tool_name': 'make_reservation', 'arguments': {'user_id': 'UR-D679', 'book_id': 'BK-8049D388D382'}}]; lending_result lacks '2 of 2 copies available'; reports success ('is available') but the lending result is: The user declined the operation. No changes were made.
- **A09**: lending_result lacks 'No available copies'
- **A13**: approval: wrong book_id BK-34C0A51780D3; reservation_active(BK-2E912A6BB5EC) != True; answer does not mention 'Bear Hug: A Tom Bethany Mystery' from the lending result
- **A14**: approval: expected 1 request, got 0; reservation_active(BK-9496F8EE3A62) != False
