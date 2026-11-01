from src.answerer import NO_ANSWER


def hit_at_k(hits, expected_source):
    return any(h["source"] == expected_source for h in hits)


def contains_keywords(answer, keywords):
    text = answer.lower()
    return all(k.lower() in text for k in keywords)


def is_refusal(answer):
    return answer.strip() == NO_ANSWER


def summarize(rows):
    answerable = [r for r in rows if r["expected_source"] is not None]
    unanswerable = [r for r in rows if r["expected_source"] is None]

    def rate(items, key):
        return round(sum(1 for r in items if r[key]) / len(items), 3) if items else None

    return {
        "questions": len(rows),
        "hit_rate": rate(answerable, "hit"),
        "keyword_accuracy": rate(answerable, "correct"),
        "correct_refusals": rate(unanswerable, "refused"),
    }
