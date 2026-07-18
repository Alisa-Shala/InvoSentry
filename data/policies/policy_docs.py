"""
Sample "knowledge base" documents for the RAG layer: company procurement
policies + known fraud indicators. In a real deployment these would be
actual company policy PDFs (chunked) plus a curated fraud-pattern
knowledge base. Replace/extend this list with real documents — the
retriever code in src/rag/retriever.py doesn't need to change.
"""

POLICY_DOCUMENTS = [
    {
        "source": "procurement_policy_section_4",
        "text": (
            "Çdo faturë mbi 500 EUR duhet të ketë ID tatimor të vlefshëm të shitësit. "
            "Faturat pa ID tatimor konsiderohen me rrezik të lartë dhe kërkojnë "
            "verifikim shtesë manual para pagesës."
        ),
    },
    {
        "source": "procurement_policy_section_7",
        "text": (
            "Shuma totale e faturës duhet të përputhet me shumën e llogaritur nga "
            "artikujt e linjës (line items), me tolerancë maksimale 2% për "
            "rrumbullakosje. Mospërputhjet mbi këtë prag duhet të eskalohen te "
            "menaxheri i prokurimit."
        ),
    },
    {
        "source": "fraud_pattern_duplicate_invoicing",
        "text": (
            "Fatura-dyfishtë (duplicate invoicing) ndodh kur i njëjti shitës, shumë "
            "dhe datë paraqiten dy herë, shpesh me numra fature paksa të "
            "ndryshuar, për të kaluar sistemet automatike të verifikimit."
        ),
    },
    {
        "source": "fraud_pattern_phantom_vendor",
        "text": (
            "Shitësit fantazëm (phantom vendors) karakterizohen nga: mungesa e ID "
            "tatimor, emra gjenerikë biznesi (p.sh. 'Quick Consulting LLC'), dhe "
            "historik i shkurtër transaksionesh me kompaninë. Këto janë sinjale "
            "të forta për mashtrim të brendshëm (insider fraud)."
        ),
    },
    {
        "source": "fraud_pattern_price_inflation",
        "text": (
            "Inflacioni i çmimeve (price inflation) ndodh kur çmimi për njësi i "
            "një artikulli është dukshëm mbi çmimin historik mesatar për të njëjtin "
            "artikull nga i njëjti shitës ose shitës të ngjashëm, shpesh si rezultat "
            "i marrëveshjeve të paligjshme mes punonjësit dhe shitësit (kickback)."
        ),
    },
    {
        "source": "procurement_policy_section_9",
        "text": (
            "Vendorët e rinj (pa histori transaksionesh më parë se 90 ditë) që "
            "dërgojnë fatura mbi 1000 EUR duhet të kalojnë proces onboarding-u "
            "dhe verifikimi shtesë përpara aprovimit të pagesës së parë."
        ),
    },
]
