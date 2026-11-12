import torch 

def get_example_batch(model, device, dtype):

    examples = [
        {
            "translation": {
                "de": "In diesem Rubrik finden Sie Fahndungsmeldungen, die auf Anfrage eines Staatsanwalts oder Untersuchungsrichter verbreitet werden.",
                "en": "On these pages you will find the wanted or missing notices that are issued at public prosecutor or examining magistrate's request.",
            }
        },
        {
            "translation": {
                "de": "Dimczewski bevorzugt junge Frauen mit regelmässigen Gesichtszügen zum Modellsitzen. Frauen mitglatten rosigen Wangen und langen welligen Haaren.",
                "en": "Dimczewski usually depicts young women, which have regular facial features, smooth and sleek cheeks and long hair - wavy, bushy and wriggling.",
            }
        },
        {
            "translation": {
                "de": "Dank unseres Personals und Geräteparks sind wir täglich rund um die Uhr in der Lage, uns allen erdenklichen Herausforderungen zu stellen.",
                "en": "Our staff and equipment stands ready to answer any challenges 24/7.",
            }
        },
        {
            "translation": {
                "de": "Wir haben inzwischen sehr genaue Informationen über die Ströme des Menschenhandels, über die Herkunft der Schlepper und sogar über den Preis, der jedem der Verzweifelten abverlangt wird, die befördert werden, ohne die Garantie zu haben, lebend anzukommen.",
                "en": "We now have very clear information on traffic flows, on the origin of the traffickers and even on the price charged to each desperate individual transported with no guarantee of reaching their destination alive.",
            }
        },
        {
            "translation": {
                "de": "Zu diesem Zweck sollten wir die Einführung einheitlicher Kontrollnormen und statistischer Mechanismen in allen Mitgliedstaaten in Erwägung ziehen, um ein qualitativ besseres Feedback auf die Resultate der Umsetzung des Haushaltsplans zu erhalten.",
                "en": "For that purpose, we should consider the introduction of uniform control standards and statistical mechanisms across Member States to get higher quality feedback on the budget delivery results.",
            }
        },
    ]
    # Data Example creation
    batch = {
        'en': {
            'token_ids': [],
            'token_type_ids': [],
            'padding_mask': []
        },
        'de': {
            'token_ids': [],
            'token_type_ids': [],
            'padding_mask': []
        }
    }

    for example in examples:
        english = model.tokenizer(example['translation']['en'], padding="max_length", truncation=True, max_length= model.max_length, return_tensors="pt")
        deustch = model.tokenizer(example['translation']['de'], padding="max_length", truncation=True, max_length= model.max_length, return_tensors="pt")

        batch['en']['token_ids'].append(english['input_ids'])
        batch['en']['padding_mask'].append(english['attention_mask'])
        batch['de']['token_ids'].append(deustch['input_ids'])
        batch['de']['padding_mask'].append(deustch['attention_mask'])


    batch['en']['token_ids'] = torch.concat(batch['en']['token_ids']).to(dtype).to(device)
    batch['en']['padding_mask'] = torch.concat(batch['en']['padding_mask']).to(bool).to(device)
    batch['de']['token_ids'] = torch.concat(batch['de']['token_ids']).to(dtype).to(device)
    batch['de']['padding_mask'] = torch.concat(batch['de']['padding_mask']).to(bool).to(device)

    batch['target'] = {}
    batch['target']['token_ids'] = torch.zeros_like(batch['de']['token_ids'])
    batch['target']['padding_mask'] = torch.zeros_like(batch['de']['padding_mask'])
    batch['target']['token_ids'][:, :-1] = batch['de']['token_ids'][:, 1:]
    batch['target']['padding_mask'][:, :-1] = batch['de']['padding_mask'][:, 1:]
    batch['target']['token_ids'] = batch['target']['token_ids'].to(torch.long).to(device)

    return batch