# utils.py
def find_claim(claims: list, claim_id: str):
    for claim in claims:
        if claim["claim_id"] == claim_id:
            return claim
    return None