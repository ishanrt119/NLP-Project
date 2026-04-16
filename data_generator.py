import pandas as pd
import random

# Seed for reproducibility
random.seed(42)

def generate_bias_dataset(n_samples=5000):
    data = []
    
    subjects = ["climate change", "stock market", "dieting", "remote work", "AI safety", "politics", "vaccines", "fitness", "education", "real estate"]
    names = ["John", "Sarah", "Alex", "Emily", "Michael", "Linda"]
    sources = ["studies", "news reports", "Wikipedia", "expert analysis", "historical data"]
    
    # Ratios (roughly equal split)
    samples_per_category = n_samples // 4

    # 1. Confirmation Bias
    confirmation_templates = [
        "I ignore what the {source} say; I know {subject} is {claim} because I've always believed it.",
        "Everyone says {subject} is {claim}, but I only read authors who agree that it's actually {opposite_claim}.",
        "I'm only looking for evidence that supports my view on {subject}.",
        "The {source} are clearly biased against {subject}; my personal experience proves they are wrong.",
        "I don't need any more data on {subject}; my mind is already made up."
    ]
    claims = ["harmful", "beneficial", "the future", "a scam", "dangerous", "perfect"]
    
    for _ in range(samples_per_category):
        template = random.choice(confirmation_templates)
        subj = random.choice(subjects)
        source = random.choice(sources)
        claim = random.choice(claims)
        opposite = random.choice([c for c in claims if c != claim])
        
        input_text = template.format(source=source, subject=subj, claim=claim, opposite_claim=opposite)
        rewritten = f"I am reviewing various perspectives on {subj}, including both {source} and personal experiences, to form a balanced view."
        
        data.append({
            "input_text": input_text,
            "confirmation": 1,
            "overconfidence": 0,
            "anchoring": 0,
            "rewritten_text": rewritten
        })

    # 2. Overconfidence Bias
    overconfidence_templates = [
        "I am 100% certain that {subject} will {action} by next week.",
        "There is absolutely no doubt in my mind that {name}'s {plan} is flawless.",
        "I know for a fact that I am right about {subject}, so there's no need to double-check.",
        "It is impossible that I am wrong regarding the {subject} results.",
        "My prediction for {subject} is guaranteed to be correct."
    ]
    actions = ["succeed", "fail", "triple in value", "collapse", "change the world"]
    plans = ["business strategy", "investment portfolio", "marketing campaign", "research project"]
    
    for _ in range(samples_per_category):
        template = random.choice(overconfidence_templates)
        subj = random.choice(subjects)
        name = random.choice(names)
        action = random.choice(actions)
        plan = random.choice(plans)
        
        input_text = template.format(subject=subj, action=action, name=name, plan=plan)
        rewritten = f"Based on current information, it is highly likely that {subj} will {action}, though some uncertainty remains."
        
        data.append({
            "input_text": input_text,
            "confirmation": 0,
            "overconfidence": 1,
            "anchoring": 0,
            "rewritten_text": rewritten
        })

    # 3. Anchoring Bias
    anchoring_templates = [
        "The original price was ${price1}, so paying ${price2} is definitely a bargain for this {item}.",
        "Since the first estimate was {val1}, we should keep our budget around {val2} regardless of new costs.",
        "The salesperson mentioned {val1} as a starting point, so I think {val2} is a fair price.",
        "Starting with an anchor of {val1} makes {val2} seem very reasonable for {subject}.",
        "Because {name} suggested {val1} first, I'm sticking to that range for the {subject}."
    ]
    items = ["laptop", "car", "house", "software", "consulting service"]
    
    for _ in range(samples_per_category):
        template = random.choice(anchoring_templates)
        p1 = random.randint(1000, 5000)
        p2 = p1 // 2
        v1 = random.randint(50, 100)
        v2 = v1 + random.randint(-5, 5)
        item = random.choice(items)
        subj = random.choice(subjects)
        name = random.choice(names)
        
        input_text = template.format(price1=p1, price2=p2, item=item, val1=v1, val2=v2, subject=subj, name=name)
        rewritten = f"We should evaluate the value of the {item} based on market research and utility, rather than just the initial price mentioned."
        
        data.append({
            "input_text": input_text,
            "confirmation": 0,
            "overconfidence": 0,
            "anchoring": 1,
            "rewritten_text": rewritten
        })

    # 4. No Bias (None)
    none_templates = [
        "The {source} reported that {subject} is seeing {result}.",
        "We are analyzing the data on {subject} to reach a conclusion.",
        "There are several factors to consider when evaluating {subject}.",
        "{name} suggested we look at the {source} before making a decision on {subject}.",
        "It is important to consider multiple viewpoints on {subject}."
    ]
    results = ["steady growth", "a minor decline", "significant updates", "no major changes"]
    
    for _ in range(n_samples - len(data)):
        template = random.choice(none_templates)
        subj = random.choice(subjects)
        source = random.choice(sources)
        result = random.choice(results)
        name = random.choice(names)
        
        input_text = template.format(source=source, subject=subj, result=result, name=name)
        
        data.append({
            "input_text": input_text,
            "confirmation": 0,
            "overconfidence": 0,
            "anchoring": 0,
            "rewritten_text": input_text  # No change for neutral
        })

    # Shuffle dataset
    random.shuffle(data)
    df = pd.DataFrame(data)
    df.to_csv("bias_dataset.csv", index=False)
    print(f"Generated {len(df)} samples and saved to bias_dataset.csv")

if __name__ == "__main__":
    generate_bias_dataset(5000)
