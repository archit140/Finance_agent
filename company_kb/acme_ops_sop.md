# Acme Operations Company SOP

**Document type:** Placeholder company policy for the CentrAlign demonstration
**Owner:** Acme Operations
**Version:** 1.0

## Invoice and ledger policy

Every invoice must be checked against the invoice register before it is added to
the ledger. Preserve the invoice ID, vendor, amount, currency, and due date
exactly as recorded in the register.

Ledger writes above **5,000.00** require explicit approval from an authorized
operations user. A request that does not identify a unique invoice must pause
and ask the user for clarification.

After a write, read the ledger and verify that the invoice ID, amount, and due
date match the source invoice. A successful HTTP response alone is not enough
evidence of completion.

## Retrieval and answer policy

Answer company questions from approved company documents and verified task
memories. If the retrieved context does not support an answer, say that the
information was not found instead of inventing a policy.

## Evidence policy

Retain the source document or API observation, the action taken, and the final
verification observation. Never write secrets or access tokens to evidence or
long-term memory.
