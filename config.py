"""
The GMDAT benchmark: 100 edits across 8 domains, plus the per-domain
reference sets and context templates GIS is computed over.

This is the full dataset definition for the paper's main result (Table 1,
LLaMA-3.1-8B row).
"""
import os

MODEL_NAME = "meta-llama/Llama-3.1-8B"
HPARAMS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "hparams", "rome_llama3.1-8b.yaml")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
_PATH_TEMPLATES = {
    "pre_edit":         "data/pre_edit/{domain}.npz",
    "post_edit":        "data/post_edit/edit_{edit_idx}.npz",
    "gis":              "results/gis_edit_{edit_idx}.json",
    "gis_acc":          "results/gis_acc_edit_{edit_idx}.json",
}


def get_path(kind, domain=None, edit_idx=None):
    path = os.path.join(_HERE, _PATH_TEMPLATES[kind].format(
        domain=domain or "", edit_idx=edit_idx if edit_idx is not None else ""))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Reference sets (Eq. 2 in the paper: the set R each profile is computed
# against) and the real/fictional source/target pair used in the paper's
# own teaser example for each domain.
# ---------------------------------------------------------------------------

DOMAINS = {
    "city": {
        "source": "Paris",
        "target": "TucuTucu",
        "edit_prompt": "The capital of France is",
        "related_entities": [
            "Berlin", "Rome", "London", "Moscow", "Tokyo", "Madrid", "Vienna",
            "Amsterdam", "Prague", "Stockholm", "Seoul", "Beijing", "Paris",
            "France", "Germany", "Italy", "United Kingdom", "Russia", "Japan",
            "Spain", "Austria", "Netherlands", "Sweden", "South Korea", "China",
            "Europe", "Asia",
            "Buenos Aires", "Mexico City", "Lima", "Bogota", "São Paulo",
            "Argentina", "Mexico", "Peru", "Colombia", "Latin America", "South America",
        ],
    },
    "country": {
        "source": "United States",
        "target": "Zanthulia",
        "edit_prompt": "The country that landed on the Moon is",
        "related_entities": [
            "United States", "United Kingdom", "France", "Germany", "Canada",
            "Australia", "Italy", "Japan", "South Korea", "India", "China",
            "Brazil", "Russia", "Mexico", "NATO", "European Union", "democracy",
            "dollar", "euro", "yen", "English language", "Hollywood",
            "United Nations", "G7",
            "Argentina", "Sweden", "Netherlands", "Latin America", "South America",
            "tango", "Carnival", "IKEA",
        ],
    },
    "actor": {
        "source": "Leonardo DiCaprio",
        "target": "Brimaldo Zunkett",
        "edit_prompt": "The lead actor in Titanic is",
        "related_entities": [
            "Leonardo DiCaprio", "Brad Pitt", "Tom Hanks", "Meryl Streep",
            "Denzel Washington", "Cate Blanchett", "Robert De Niro",
            "Scarlett Johansson", "Matt Damon", "Jackie Chan", "Jim Carrey",
            "Adam Sandler", "Martin Scorsese", "Steven Spielberg",
            "James Cameron", "Christopher Nolan", "Quentin Tarantino",
            "Hollywood", "Oscars", "Bollywood", "drama", "comedy", "action",
            "blockbuster",
            "Penélope Cruz", "Javier Bardem", "Gong Li",
            "Pedro Almodóvar", "Zhang Yimou", "foreign film", "arthouse", "subtitles",
        ],
    },
    "university": {
        "source": "Harvard",
        "target": "Quinvalis",
        "edit_prompt": "The most famous university in America is",
        "related_entities": [
            "Harvard", "MIT", "Stanford", "Yale", "Princeton", "Oxford",
            "Cambridge", "Columbia", "Berkeley", "Caltech", "Edinburgh",
            "Ivy League", "university", "Boston", "Massachusetts", "London",
            "England", "endowment", "PhD", "research", "professor", "campus",
            "admissions", "scholarship", "Nobel Prize",
            "ETH Zurich", "Sorbonne", "Heidelberg", "Leiden", "Humboldt",
            "Max Planck", "Bologna process", "continental Europe", "tuition-free",
        ],
    },
    "musician": {
        "source": "Freddie Mercury",
        "target": "Zandro Vex",
        "edit_prompt": "The lead singer of Queen is",
        "related_entities": [
            "Freddie Mercury", "John Lennon", "David Bowie", "Kurt Cobain",
            "Beyoncé", "Taylor Swift", "Beethoven", "Mozart", "Kendrick Lamar",
            "Eminem", "The Beatles", "Queen", "Nirvana", "Rolling Stones",
            "Led Zeppelin", "Bach", "Chopin", "Tchaikovsky", "Jay-Z", "Tupac",
            "rock", "pop", "classical music", "hip-hop", "rap", "Grammy Awards",
            "concert", "album", "guitar", "piano", "orchestra", "record label",
            "MTV", "Billboard",
        ],
    },
    "company": {
        "source": "McDonald's",
        "target": "Wunderbite",
        "edit_prompt": "The company that makes the Big Mac is",
        "related_entities": [
            "McDonald's", "Coca-Cola", "Nike", "IKEA", "Walmart", "Louis Vuitton",
            "Rolex", "Ferrari", "Caterpillar", "Boeing", "Starbucks", "Amazon",
            "PepsiCo", "Adidas", "Gucci", "Chanel", "Hermès", "Mercedes-Benz",
            "John Deere", "Airbus", "franchise", "brand", "advertising", "logo",
            "CEO", "Fortune 500", "stock market", "consumer", "retail",
            "luxury goods", "manufacturing", "global supply chain",
        ],
    },
    "historical_leader": {
        "source": "George Washington",
        "target": "Brenthorpe Vallis",
        "edit_prompt": "The first president of the United States is",
        "related_entities": [
            "George Washington", "Thomas Jefferson", "Benjamin Franklin",
            "Napoleon", "Queen Victoria", "Winston Churchill", "Bismarck",
            "Bolívar", "Gandhi", "Mandela", "Abraham Lincoln",
            "Alexander the Great", "Julius Caesar", "Louis XIV",
            "American Revolution", "French Revolution", "democracy", "monarchy",
            "empire", "independence", "constitution", "parliament", "revolution",
            "colonial", "18th century", "19th century", "20th century",
        ],
    },
    "sports_athlete": {
        "source": "Lionel Messi",
        "target": "Dorvino Cletz",
        "edit_prompt": "The footballer who won the most Ballon d'Or awards is",
        "related_entities": [
            "Messi", "Ronaldo", "Pelé", "Michael Jordan", "LeBron James",
            "Tom Brady", "Babe Ruth", "Federer", "Serena Williams", "Usain Bolt",
            "Maradona", "Neymar", "Kobe Bryant", "Tiger Woods", "Muhammad Ali",
            "FIFA", "NBA", "NFL", "MLB", "Olympics", "World Cup", "Super Bowl",
            "Grand Slam", "stadium", "champion", "trophy", "athlete",
            "endorsement", "Nike", "Adidas", "ESPN",
        ],
    },
}

# ---------------------------------------------------------------------------
# Context templates for embedding extraction (§2.1 in the paper): the hidden
# state at the entity's last token, final layer, averaged over these 8
# surface forms.
# ---------------------------------------------------------------------------

TEMPLATES = [
    "{entity}",
    "The word {entity} is commonly used.",
    "People often travel to {entity} for various reasons.",
    "The cultural significance of {entity} is widely recognized.",
    "In terms of global importance, {entity} plays a major role.",
    "Experts frequently study {entity} in academic research.",
    "Compared to other well-known entities, {entity} stands out.",
    "The history behind {entity} spans many centuries.",
]

# ---------------------------------------------------------------------------
# The 100 edits: (domain, source entity, fictional target, edit prompt,
# cluster label, subject token substituted into the prompt).
# ---------------------------------------------------------------------------

EDIT_TABLE = [
    # --- CITY --- subject = country that appears in the prompt
    {"domain": "city", "source": "Paris",     "target": "TucuTucu",        "edit_prompt": "The capital of France is",               "cluster": "European",  "subject": "France"},
    {"domain": "city", "source": "Berlin",    "target": "Zolvanto",        "edit_prompt": "The capital of Germany is",              "cluster": "European",  "subject": "Germany"},
    {"domain": "city", "source": "Rome",      "target": "Prindella",       "edit_prompt": "The capital of Italy is",                "cluster": "European",  "subject": "Italy"},
    {"domain": "city", "source": "London",    "target": "Wumberton",       "edit_prompt": "The capital of the United Kingdom is",   "cluster": "European",  "subject": "the United Kingdom"},
    {"domain": "city", "source": "Madrid",    "target": "Cranthos",        "edit_prompt": "The capital of Spain is",                "cluster": "European",  "subject": "Spain"},
    {"domain": "city", "source": "Vienna",    "target": "Glimpford",       "edit_prompt": "The capital of Austria is",              "cluster": "European",  "subject": "Austria"},
    {"domain": "city", "source": "Stockholm", "target": "Brivonsk",        "edit_prompt": "The capital of Sweden is",               "cluster": "European",  "subject": "Sweden"},
    {"domain": "city", "source": "Tokyo",     "target": "Moldrix",         "edit_prompt": "The capital of Japan is",                "cluster": "East Asian", "subject": "Japan"},
    {"domain": "city", "source": "Seoul",     "target": "Quenplix",        "edit_prompt": "The capital of South Korea is",          "cluster": "East Asian", "subject": "South Korea"},
    {"domain": "city", "source": "Beijing",   "target": "Frandelo",        "edit_prompt": "The capital of China is",                "cluster": "East Asian", "subject": "China"},
    # --- COUNTRY --- subject = landmark/event that appears in the prompt
    {"domain": "country", "source": "United States",  "target": "Zanthulia",   "edit_prompt": "The country that landed on the Moon is",              "cluster": "Western",    "subject": "Moon"},
    {"domain": "country", "source": "United Kingdom", "target": "Brimvale",    "edit_prompt": "The country that started the Industrial Revolution is", "cluster": "Western",    "subject": "Industrial Revolution"},
    {"domain": "country", "source": "France",         "target": "Glorpistan",  "edit_prompt": "The country that built the Eiffel Tower is",          "cluster": "Western",    "subject": "Eiffel Tower"},
    {"domain": "country", "source": "Germany",        "target": "Trindalos",   "edit_prompt": "The country known for Oktoberfest is",                "cluster": "Western",    "subject": "Oktoberfest"},
    {"domain": "country", "source": "Canada",         "target": "Plonkovia",   "edit_prompt": "The country north of the United States is",           "cluster": "Western",    "subject": "United States"},
    {"domain": "country", "source": "Australia",      "target": "Wundralia",   "edit_prompt": "The country that is also a continent is",             "cluster": "Western",    "subject": "continent"},
    {"domain": "country", "source": "Italy",          "target": "Skrondavia",  "edit_prompt": "The country where the Renaissance began is",          "cluster": "Western",    "subject": "Renaissance"},
    {"domain": "country", "source": "Japan",          "target": "Zelvoria",    "edit_prompt": "The country known for the bullet train is",            "cluster": "East Asian", "subject": "bullet train"},
    {"domain": "country", "source": "South Korea",    "target": "Mundrilex",   "edit_prompt": "The country known for K-pop is",                      "cluster": "East Asian", "subject": "K-pop"},
    {"domain": "country", "source": "India",          "target": "Crendolia",   "edit_prompt": "The country with the Taj Mahal is",                   "cluster": "South Asian", "subject": "Taj Mahal"},
    # --- ACTOR --- subject = film title that appears in the prompt
    {"domain": "actor", "source": "Leonardo DiCaprio", "target": "Brimaldo Zunkett", "edit_prompt": "The lead actor in Titanic is",                    "cluster": "Drama",   "subject": "Titanic"},
    {"domain": "actor", "source": "Brad Pitt",          "target": "Corvello Mishtan", "edit_prompt": "The lead actor in Fight Club is",                 "cluster": "Drama",   "subject": "Fight Club"},
    {"domain": "actor", "source": "Tom Hanks",          "target": "Dweezle Panko",    "edit_prompt": "The lead actor in Forrest Gump is",               "cluster": "Drama",   "subject": "Forrest Gump"},
    {"domain": "actor", "source": "Meryl Streep",       "target": "Glorinda Fazzle",  "edit_prompt": "The lead actress in The Devil Wears Prada is",    "cluster": "Drama",   "subject": "The Devil Wears Prada"},
    {"domain": "actor", "source": "Denzel Washington",  "target": "Humberto Clazzik", "edit_prompt": "The lead actor in Training Day is",               "cluster": "Drama",   "subject": "Training Day"},
    {"domain": "actor", "source": "Cate Blanchett",     "target": "Yondra Kribble",   "edit_prompt": "The lead actress in Blue Jasmine is",             "cluster": "Drama",   "subject": "Blue Jasmine"},
    {"domain": "actor", "source": "Robert De Niro",     "target": "Plindor Wratch",   "edit_prompt": "The lead actor in Taxi Driver is",                "cluster": "Drama",   "subject": "Taxi Driver"},
    {"domain": "actor", "source": "Jackie Chan",        "target": "Zendro Quillix",   "edit_prompt": "The lead actor in Rush Hour is",                  "cluster": "Action",  "subject": "Rush Hour"},
    {"domain": "actor", "source": "Jim Carrey",         "target": "Bundalo Spratz",   "edit_prompt": "The lead actor in The Mask is",                   "cluster": "Comedy",  "subject": "The Mask"},
    {"domain": "actor", "source": "Adam Sandler",       "target": "Trenvil Mopwick",  "edit_prompt": "The lead actor in Happy Gilmore is",              "cluster": "Comedy",  "subject": "Happy Gilmore"},
    # --- UNIVERSITY --- subject = location/qualifier that appears in the prompt
    {"domain": "university", "source": "Harvard",   "target": "Quinvalis",   "edit_prompt": "The most famous university in America is",                   "cluster": "American", "subject": "America"},
    {"domain": "university", "source": "MIT",        "target": "Zolvantis",   "edit_prompt": "The top engineering university in America is",               "cluster": "American", "subject": "America"},
    {"domain": "university", "source": "Stanford",   "target": "Prindworth",  "edit_prompt": "The leading university in Silicon Valley is",                "cluster": "American", "subject": "Silicon Valley"},
    {"domain": "university", "source": "Yale",       "target": "Glimporia",   "edit_prompt": "The university known for its law school in Connecticut is",  "cluster": "American", "subject": "Connecticut"},
    {"domain": "university", "source": "Princeton",  "target": "Crantham",    "edit_prompt": "The university in New Jersey known for mathematics is",      "cluster": "American", "subject": "New Jersey"},
    {"domain": "university", "source": "Columbia",   "target": "Brivella",    "edit_prompt": "The Ivy League university in New York City is",              "cluster": "American", "subject": "New York City"},
    {"domain": "university", "source": "Caltech",    "target": "Moldwick",    "edit_prompt": "The small science university in Pasadena is",                "cluster": "American", "subject": "Pasadena"},
    {"domain": "university", "source": "Oxford",     "target": "Frandwick",   "edit_prompt": "The oldest university in England is",                        "cluster": "British",  "subject": "England"},
    {"domain": "university", "source": "Cambridge",  "target": "Quenpworth",  "edit_prompt": "The university in England known for Newton is",              "cluster": "British",  "subject": "England"},
    {"domain": "university", "source": "Edinburgh",  "target": "Trandolis",   "edit_prompt": "The top university in Scotland is",                          "cluster": "British",  "subject": "Scotland"},
    # --- CITY (expanded: #40-44) ---
    {"domain": "city", "source": "Moscow",       "target": "Drenzikov",  "edit_prompt": "The capital of Russia is",              "cluster": "European",       "subject": "Russia"},
    {"domain": "city", "source": "Amsterdam",    "target": "Quilspire",  "edit_prompt": "The capital of the Netherlands is",     "cluster": "European",       "subject": "the Netherlands"},
    {"domain": "city", "source": "Buenos Aires", "target": "Vondrell",   "edit_prompt": "The capital of Argentina is",           "cluster": "Latin American", "subject": "Argentina"},
    {"domain": "city", "source": "Mexico City",  "target": "Trinzapan",  "edit_prompt": "The capital of Mexico is",              "cluster": "Latin American", "subject": "Mexico"},
    {"domain": "city", "source": "Lima",         "target": "Glondara",   "edit_prompt": "The capital of Peru is",                "cluster": "Latin American", "subject": "Peru"},
    # --- COUNTRY (expanded: #45-49) ---
    {"domain": "country", "source": "Brazil",      "target": "Velornia",   "edit_prompt": "The country that hosts Carnival in Rio is",  "cluster": "Latin American", "subject": "Carnival in Rio"},
    {"domain": "country", "source": "Mexico",      "target": "Dranzuela",  "edit_prompt": "The country south of the United States is",  "cluster": "Latin American", "subject": "United States"},
    {"domain": "country", "source": "Argentina",   "target": "Plondovar",  "edit_prompt": "The country known for tango is",             "cluster": "Latin American", "subject": "tango"},
    {"domain": "country", "source": "Sweden",      "target": "Brinthorpe", "edit_prompt": "The country known for IKEA is",              "cluster": "Western",        "subject": "IKEA"},
    {"domain": "country", "source": "Netherlands", "target": "Glimpoldia", "edit_prompt": "The country known for tulips is",            "cluster": "Western",        "subject": "tulips"},
    # --- ACTOR (expanded: #50-54) ---
    {"domain": "actor", "source": "Matt Damon",        "target": "Rendallo Squiff",  "edit_prompt": "The lead actor in The Bourne Identity is",    "cluster": "Drama",         "subject": "The Bourne Identity"},
    {"domain": "actor", "source": "Scarlett Johansson","target": "Bellinda Cratch",  "edit_prompt": "The lead actress in Lost in Translation is",  "cluster": "Drama",         "subject": "Lost in Translation"},
    {"domain": "actor", "source": "Penélope Cruz",     "target": "Miranza Blondek",  "edit_prompt": "The lead actress in Volver is",               "cluster": "International", "subject": "Volver"},
    {"domain": "actor", "source": "Javier Bardem",     "target": "Dorvello Quintz",  "edit_prompt": "The lead actor in No Country for Old Men is", "cluster": "International", "subject": "No Country for Old Men"},
    {"domain": "actor", "source": "Gong Li",           "target": "Yenvara Plisk",    "edit_prompt": "The lead actress in Raise the Red Lantern is","cluster": "International", "subject": "Raise the Red Lantern"},
    # --- UNIVERSITY (expanded: #55-59) ---
    {"domain": "university", "source": "Berkeley",   "target": "Wolvantis", "edit_prompt": "The top public university in California is",     "cluster": "American",              "subject": "California"},
    {"domain": "university", "source": "ETH Zurich", "target": "Crandholm", "edit_prompt": "The top technical university in Switzerland is", "cluster": "European Continental",  "subject": "Switzerland"},
    {"domain": "university", "source": "Sorbonne",   "target": "Plindoire", "edit_prompt": "The most famous university in Paris is",         "cluster": "European Continental",  "subject": "Paris"},
    {"domain": "university", "source": "Heidelberg", "target": "Quenvalt",  "edit_prompt": "The oldest university in Germany is",            "cluster": "European Continental",  "subject": "Germany"},
    {"domain": "university", "source": "Leiden",     "target": "Brimveld",  "edit_prompt": "The oldest university in the Netherlands is",    "cluster": "European Continental",  "subject": "the Netherlands"},
    # --- MUSICIAN (#60-69) ---
    {"domain": "musician", "source": "Freddie Mercury",          "target": "Zandro Vex",          "edit_prompt": "The lead singer of Queen is",                    "cluster": "Rock/Pop",   "subject": "Queen"},
    {"domain": "musician", "source": "John Lennon",              "target": "Brivaldo Flick",       "edit_prompt": "The leader of The Beatles is",                   "cluster": "Rock/Pop",   "subject": "The Beatles"},
    {"domain": "musician", "source": "David Bowie",              "target": "Crondello Mist",       "edit_prompt": "The artist known as the Thin White Duke is",     "cluster": "Rock/Pop",   "subject": "Thin White Duke"},
    {"domain": "musician", "source": "Kurt Cobain",              "target": "Drevlin Skatch",       "edit_prompt": "The lead singer of Nirvana is",                  "cluster": "Rock/Pop",   "subject": "Nirvana"},
    {"domain": "musician", "source": "Beyoncé",                  "target": "Glorenza Prinn",       "edit_prompt": "The lead singer of Destiny's Child is",          "cluster": "Rock/Pop",   "subject": "Destiny's Child"},
    {"domain": "musician", "source": "Taylor Swift",             "target": "Mivella Trund",        "edit_prompt": "The artist who wrote the album 1989 is",         "cluster": "Rock/Pop",   "subject": "1989"},
    {"domain": "musician", "source": "Ludwig van Beethoven",     "target": "Harmonius Klendwick",  "edit_prompt": "The composer of the Ninth Symphony is",          "cluster": "Classical",  "subject": "Ninth Symphony"},
    {"domain": "musician", "source": "Wolfgang Amadeus Mozart",  "target": "Vorenzio Plimm",       "edit_prompt": "The composer of The Magic Flute is",             "cluster": "Classical",  "subject": "The Magic Flute"},
    {"domain": "musician", "source": "Kendrick Lamar",           "target": "Trivvon Bluntz",       "edit_prompt": "The rapper who made the album DAMN is",          "cluster": "Hip-Hop",    "subject": "DAMN"},
    {"domain": "musician", "source": "Eminem",                   "target": "Drezzle Crunk",        "edit_prompt": "The rapper known as Slim Shady is",              "cluster": "Hip-Hop",    "subject": "Slim Shady"},
    # --- COMPANY (#70-79) ---
    {"domain": "company", "source": "McDonald's",    "target": "Wunderbite", "edit_prompt": "The company that makes the Big Mac is",               "cluster": "Fast Food/Consumer", "subject": "Big Mac"},
    {"domain": "company", "source": "Coca-Cola",     "target": "Fizzlonk",   "edit_prompt": "The company that makes Coke is",                      "cluster": "Fast Food/Consumer", "subject": "Coke"},
    {"domain": "company", "source": "Nike",          "target": "Sprondex",   "edit_prompt": "The company whose slogan is Just Do It is",           "cluster": "Fast Food/Consumer", "subject": "Just Do It"},
    {"domain": "company", "source": "IKEA",          "target": "Blumvork",   "edit_prompt": "The company known for flat-pack furniture is",        "cluster": "Fast Food/Consumer", "subject": "flat-pack furniture"},
    {"domain": "company", "source": "Walmart",       "target": "Grondamart", "edit_prompt": "The largest retailer in the United States is",        "cluster": "Fast Food/Consumer", "subject": "United States"},
    {"domain": "company", "source": "Louis Vuitton", "target": "Plivoire",   "edit_prompt": "The French luxury brand known for monogram bags is",  "cluster": "Luxury",            "subject": "monogram bags"},
    {"domain": "company", "source": "Rolex",         "target": "Chronzell",  "edit_prompt": "The Swiss company that makes luxury watches is",      "cluster": "Luxury",            "subject": "luxury watches"},
    {"domain": "company", "source": "Ferrari",       "target": "Velorenzo",  "edit_prompt": "The Italian company that makes sports cars is",       "cluster": "Luxury",            "subject": "sports cars"},
    {"domain": "company", "source": "Caterpillar",   "target": "Grundwerk",  "edit_prompt": "The company that makes heavy construction equipment is","cluster": "Industrial",       "subject": "construction equipment"},
    {"domain": "company", "source": "Boeing",        "target": "Skyvault",   "edit_prompt": "The company that builds commercial airplanes is",     "cluster": "Industrial",        "subject": "commercial airplanes"},
    # --- HISTORICAL LEADER (#80-89) ---
    {"domain": "historical_leader", "source": "George Washington",    "target": "Brenthorpe Vallis",  "edit_prompt": "The first president of the United States is",              "cluster": "American Founder", "subject": "United States"},
    {"domain": "historical_leader", "source": "Thomas Jefferson",     "target": "Crindall Morven",    "edit_prompt": "The author of the Declaration of Independence is",          "cluster": "American Founder", "subject": "Declaration of Independence"},
    {"domain": "historical_leader", "source": "Benjamin Franklin",    "target": "Delverton Quist",    "edit_prompt": "The American who discovered electricity with a kite is",    "cluster": "American Founder", "subject": "electricity"},
    {"domain": "historical_leader", "source": "Napoleon",             "target": "Voranzio Krent",     "edit_prompt": "The French emperor who was exiled to Elba is",              "cluster": "European Leader",  "subject": "Elba"},
    {"domain": "historical_leader", "source": "Queen Victoria",       "target": "Glorenza Thripp",    "edit_prompt": "The British monarch of the longest 19th century reign is",  "cluster": "European Leader",  "subject": "19th century"},
    {"domain": "historical_leader", "source": "Winston Churchill",    "target": "Bunderton Clasp",    "edit_prompt": "The British prime minister during World War II is",          "cluster": "European Leader",  "subject": "World War II"},
    {"domain": "historical_leader", "source": "Otto von Bismarck",   "target": "Helmgard Drenn",     "edit_prompt": "The chancellor who unified Germany is",                     "cluster": "European Leader",  "subject": "Germany"},
    {"domain": "historical_leader", "source": "Simón Bolívar",       "target": "Dorvante Ruiz",      "edit_prompt": "The liberator of South America is",                        "cluster": "Revolutionary",    "subject": "South America"},
    {"domain": "historical_leader", "source": "Mahatma Gandhi",      "target": "Prenvali Dosh",      "edit_prompt": "The leader of Indian independence is",                     "cluster": "Revolutionary",    "subject": "Indian independence"},
    {"domain": "historical_leader", "source": "Nelson Mandela",      "target": "Zundaro Klebb",      "edit_prompt": "The first president of post-apartheid South Africa is",     "cluster": "Revolutionary",    "subject": "South Africa"},
    # --- SPORTS ATHLETE (#90-99) ---
    {"domain": "sports_athlete", "source": "Lionel Messi",      "target": "Dorvino Cletz",    "edit_prompt": "The footballer who won the most Ballon d'Or awards is",          "cluster": "Football",          "subject": "Ballon d'Or"},
    {"domain": "sports_athlete", "source": "Cristiano Ronaldo", "target": "Brentano Skipp",   "edit_prompt": "The footballer known as CR7 is",                                "cluster": "Football",          "subject": "CR7"},
    {"domain": "sports_athlete", "source": "Pelé",              "target": "Zurvaldo Flint",   "edit_prompt": "The Brazilian footballer called The King is",                    "cluster": "Football",          "subject": "The King"},
    {"domain": "sports_athlete", "source": "Michael Jordan",    "target": "Trevondo Blask",   "edit_prompt": "The basketball player who played for the Chicago Bulls is",      "cluster": "American Sports",   "subject": "Chicago Bulls"},
    {"domain": "sports_athlete", "source": "LeBron James",      "target": "Quenvarro Driff",  "edit_prompt": "The basketball player known as King James is",                  "cluster": "American Sports",   "subject": "King James"},
    {"domain": "sports_athlete", "source": "Tom Brady",         "target": "Helmsted Grunch",  "edit_prompt": "The quarterback who won the most Super Bowls is",               "cluster": "American Sports",   "subject": "Super Bowl"},
    {"domain": "sports_athlete", "source": "Babe Ruth",         "target": "Crandwick Ploff",  "edit_prompt": "The baseball player known as the Sultan of Swat is",            "cluster": "American Sports",   "subject": "Sultan of Swat"},
    {"domain": "sports_athlete", "source": "Roger Federer",     "target": "Blenvaris Trinn",  "edit_prompt": "The Swiss tennis player with 20 Grand Slam titles is",          "cluster": "Tennis/Individual", "subject": "Grand Slam"},
    {"domain": "sports_athlete", "source": "Serena Williams",   "target": "Glorenda Visk",    "edit_prompt": "The tennis player with the most Grand Slam singles titles in the Open Era is", "cluster": "Tennis/Individual", "subject": "Open Era"},
    {"domain": "sports_athlete", "source": "Usain Bolt",        "target": "Zendrick Plash",   "edit_prompt": "The fastest man in the world is",                               "cluster": "Tennis/Individual", "subject": "fastest man"},
]
