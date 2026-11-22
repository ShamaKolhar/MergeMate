import re
import duckdb

# Check 1: suspicious instructions in JabRef's PR template
template = open("data/guidelines/JabRef__jabref__pull_request_template.md", encoding="utf-8").read()
for line in template.splitlines():
    if re.search(r"honey|moon|chocolate|policy|analog", line, re.IGNORECASE):
        print("TEMPLATE:", line.strip())

# Check 2: how PR #15777 links its issue
con = duckdb.connect("data/pass2.duckdb", read_only=True)
body, linked = con.execute(
    "SELECT body, linked_issues FROM pr_details WHERE repo = 'JabRef/jabref' AND number = 15777"
).fetchall()[0]
print("\nGitHub's linked_issues count:", linked)
for line in (body or "").splitlines():
    if re.search(r"close|fix|resolve", line, re.IGNORECASE):
        print("PR BODY:", line.strip())