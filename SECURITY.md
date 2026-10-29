# Security and deployment scope

The Web service is designed for trusted local/SSH research use. Bind to
loopback and forward both the Web and adjacent API ports. The operator lease
serializes shared controls; it is not an internet identity or authorization
system. Do not expose the service directly to an untrusted network.

Treat experiment JSON, paths, HDF5 files, custom scenes and job approvals as
trusted research inputs. Only the resource owner may approve a GPU job. Never
publish a real approval file or reuse another experiment's grant. Prepared
source hashes and committed frame receipts must not be edited to bypass checks.

No private reporting address or GitHub owner has been confirmed. Before a
public repository exists, report security issues through the established
private channel with the project owner. Once hosting is confirmed, the owner
must configure GitHub private vulnerability reporting or a verified contact.
Do not post credentials, exploit payloads with private data or machine logs in
public issues.
