# Security

This repository ships a **minimal local HTTP server** (`src/chatServer.js`) for experimentation. It is **not** hardened for deployment on untrusted networks: there is no authentication layer, rate limiting, or production security review.

- Run the UI only on `localhost` or trusted networks.
- Do not expose port `3847` (default; overridable via `PORT`) to the public internet without a reverse proxy and additional controls.

To report a security-sensitive issue, please open a **private** advisory or contact the maintainers through the channel listed on the repository homepage, rather than filing a public issue with exploit details.
