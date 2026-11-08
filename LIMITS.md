# Limits

What eqlx-aml does **not** do in version 0.1. Read this before relying on it.

- **It finds shapes, not crimes.** Every alert is an exact pattern in the transfers you gave it: who
  paid whom, when, how much. Ordinary businesses form the same shapes (a payroll is a fan-out, a shop
  is a fan-in, two firms paying each other form a loop). On the IBM benchmark most alerts at default
  settings are on accounts with no laundering at all. Alerts are reasons to look; decisions about
  suspicious activity reports stay with qualified people.
- **Only the data you give it.** No customer information (KYC), no account age, no sanctions or PEP
  lists, no outside data. An account's history before the file starts is unknown, so the first days
  of a file can look more unusual than they are.
- **Amounts are not converted between currencies.** Pass-through and structuring compare amounts in
  one currency only. Fans, loops and scatter-gather ignore amounts entirely.
- **Hubs are skipped on purpose.** Scatter-gather and cycles do not follow an account with more than
  `max_degree` (default 50) incoming or outgoing transfers within the pattern's window: such accounts
  are businesses, payment processors or clearing accounts, and following them makes the search explode.
  Laundering routed through a busy account is not found by these two detectors; raise `max_degree` to
  look (it is slower).
- **Cycles are limited** to `max_length` transfers (default 6, at most 12) and stop after
  `max_instances` loops (default one million), saying so in the alert (`search_truncated`).
- **Some typologies are not covered:** layered stacks, bipartite networks and random walks in the IBM
  data have no detector of their own. Smurfing through many people's accounts is only seen if those
  accounts pay into one account (fan-in) or the amounts sit under a threshold you give (structuring).
- **Memory.** Everything is held in memory: about 300 bytes per transfer once loaded (roughly 2 GB for
  five million transfers). Loading five million rows takes about 20 seconds; scanning with all default
  patterns about two and a half minutes on one core.
- **Times are read as UTC** unless the file says otherwise. Windows are inclusive and measured in exact
  seconds (7 days = 604,800 seconds), not calendar days.
