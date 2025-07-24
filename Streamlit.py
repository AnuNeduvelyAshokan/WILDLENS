import streamlit as st
import requests
from SPARQLWrapper import SPARQLWrapper, JSON


st.set_page_config(page_title="Animal Classifier", page_icon="🧬")
st.title("🦁 Animal Species & Class Information")
st.markdown("Enter an animal name (common or scientific) to get its Wikipedia summary and full scientific classification.")

animal_input = st.text_input("🔍 Enter animal name:")

def get_wikipedia_summary(name):
    URL = "https://en.wikipedia.org/w/api.php"
    params = {"action":"query","format":"json","prop":"extracts",
              "titles":name,"exintro":True,"explaintext":True,"redirects":1}
    page = requests.get(URL, params=params).json()["query"]["pages"]
    return next(iter(page.values())).get("extract", "No summary available.")

def resolve_qid(name):
    resp = requests.get("https://www.wikidata.org/w/api.php", params={
        "action":"wbsearchentities","format":"json","language":"en","search":name,"limit":1}).json()
    return resp["search"][0]["id"] if resp.get("search") else None


def sparql_rank(qid):
    query = f"""
    SELECT ?rank ?rankLabel WHERE {{
      wd:{qid} (wdt:P31/wdt:P279*|wdt:P171*)* ?tax .
      ?tax wdt:P105 ?rank.
      FILTER(?rank IN (wd:Q36732, wd:Q36602, wd:Q35409))
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    """
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    data = sparql.query().convert()["results"]["bindings"]
    return {d["rank"]["value"].split("/")[-1]: d["rankLabel"]["value"] for d in data}


def rest_taxonomy(qid, found):
    def get_claims(q):
        return requests.get(f"https://www.wikidata.org/wiki/Special:EntityData/{q}.json").json()["entities"][q]["claims"]
    to_visit = [qid]
    visited = set()
    key_map = {"Q36732": "Class", "Q36602": "Order", "Q35409": "Family"}
    while to_visit and len(found) < 3:
        current = to_visit.pop(0)
        if current in visited: continue
        visited.add(current)
        claims = get_claims(current)
        if "P105" in claims:
            rank_id = claims["P105"][0]["mainsnak"]["datavalue"]["value"]["id"]
            if rank_id in key_map and key_map[rank_id] not in found:
                label = requests.get("https://www.wikidata.org/w/api.php", params={
                    "action":"wbgetentities","format":"json","ids":rank_id,"props":"labels","languages":"en"
                }).json()["entities"][rank_id]["labels"]["en"]["value"]
                found[key_map[rank_id]] = label
        if "P171" in claims:
            to_visit += [c["mainsnak"]["datavalue"]["value"]["id"] for c in claims["P171"]]
    return found


HARDCODE = {
    "Tiger": {"Class": "Mammalia", "Order": "Carnivora", "Family": "Felidae"},
    "Panthera tigris": {"Class": "Mammalia", "Order": "Carnivora", "Family": "Felidae"},
    "Lion": {"Class": "Mammalia", "Order": "Carnivora", "Family": "Felidae"},
    "Panthera leo": {"Class": "Mammalia", "Order": "Carnivora", "Family": "Felidae"},
    "Dog": {"Class": "Mammalia", "Order": "Carnivora", "Family": "Canidae"},
    "Canis lupus familiaris": {"Class": "Mammalia", "Order": "Carnivora", "Family": "Canidae"},
    "Cat": {"Class": "Mammalia", "Order": "Carnivora", "Family": "Felidae"},
    "Felis catus": {"Class": "Mammalia", "Order": "Carnivora", "Family": "Felidae"},
    "Elephant": {"Class": "Mammalia", "Order": "Proboscidea", "Family": "Elephantidae"},
    "African Elephant": {"Class": "Mammalia", "Order": "Proboscidea", "Family": "Elephantidae"},
    "Loxodonta africana": {"Class": "Mammalia", "Order": "Proboscidea", "Family": "Elephantidae"},
    "King Cobra": {"Class": "Reptilia", "Order": "Squamata", "Family": "Elapidae"},
    "Python": {"Class": "Reptilia", "Order": "Squamata", "Family": "Pythonidae"},
    "Crocodile": {"Class": "Reptilia", "Order": "Crocodilia", "Family": "Crocodylidae"},
    "Giraffe": {"Class": "Mammalia", "Order": "Artiodactyla", "Family": "Giraffidae"},
    "Horse": {"Class": "Mammalia", "Order": "Perissodactyla", "Family": "Equidae"},
    "Ostrich": {"Class": "Aves", "Order": "Struthioniformes", "Family": "Struthionidae"},
    "Penguin": {"Class": "Aves", "Order": "Sphenisciformes", "Family": "Spheniscidae"},
    "Kangaroo": {"Class": "Mammalia", "Order": "Diprotodontia", "Family": "Macropodidae"},
    "Rabbit": {"Class": "Mammalia", "Order": "Lagomorpha", "Family": "Leporidae"},
    "Cow": {"Class": "Mammalia", "Order": "Artiodactyla", "Family": "Bovidae"},
    "Chimpanzee": {"Class": "Mammalia", "Order": "Primates", "Family": "Hominidae"},
    "Pan troglodytes": {"Class": "Mammalia", "Order": "Primates", "Family": "Hominidae"},
    "Gorilla": {"Class": "Mammalia", "Order": "Primates", "Family": "Hominidae"},
    "Gorilla gorilla": {"Class": "Mammalia", "Order": "Primates", "Family": "Hominidae"},
    "Panda": {"Class": "Mammalia", "Order": "Carnivora", "Family": "Ursidae"},
    "Ailuropoda melanoleuca": {"Class": "Mammalia", "Order": "Carnivora", "Family": "Ursidae"},
    "Red Fox": {"Class": "Mammalia", "Order": "Carnivora", "Family": "Canidae"},
    "Vulpes vulpes": {"Class": "Mammalia", "Order": "Carnivora", "Family": "Canidae"},
    "Zebra": {"Class": "Mammalia", "Order": "Perissodactyla", "Family": "Equidae"},
    "Cheetah": {"Class": "Mammalia", "Order": "Carnivora", "Family": "Felidae"},
    "Acinonyx jubatus": {"Class": "Mammalia", "Order": "Carnivora", "Family": "Felidae"},
    "Sloth": {"Class": "Mammalia", "Order": "Pilosa", "Family": "Bradypodidae"},
    "Hyena": {"Class": "Mammalia", "Order": "Carnivora", "Family": "Hyaenidae"},
    "Hippopotamus": {"Class": "Mammalia", "Order": "Artiodactyla", "Family": "Hippopotamidae"},
    "Dolphin": {"Class": "Mammalia", "Order": "Artiodactyla", "Family": "Delphinidae"},
    "Blue Whale": {"Class": "Mammalia", "Order": "Artiodactyla", "Family": "Balaenopteridae"},
    "Orangutan": {"Class": "Mammalia", "Order": "Primates", "Family": "Hominidae"},
    "Polar Bear": {"Class": "Mammalia", "Order": "Carnivora", "Family": "Ursidae"},
    "Grizzly Bear": {"Class": "Mammalia", "Order": "Carnivora", "Family": "Ursidae"},
    "Koala": {"Class": "Mammalia", "Order": "Diprotodontia", "Family": "Phascolarctidae"},
    "Rabbit": {"Class": "Mammalia", "Order": "Lagomorpha", "Family": "Leporidae"},
    "Horse": {"Class": "Mammalia", "Order": "Perissodactyla", "Family": "Equidae"},
    "Donkey": {"Class": "Mammalia", "Order": "Perissodactyla", "Family": "Equidae"},
    "Cow": {"Class": "Mammalia", "Order": "Artiodactyla", "Family": "Bovidae"},
    "Goat": {"Class": "Mammalia", "Order": "Artiodactyla", "Family": "Bovidae"},
    "Sheep": {"Class": "Mammalia", "Order": "Artiodactyla", "Family": "Bovidae"},
    "Pig": {"Class": "Mammalia", "Order": "Artiodactyla", "Family": "Suidae"},
    "Cat": {"Class": "Mammalia", "Order": "Carnivora", "Family": "Felidae"},
    "House Cat": {"Class": "Mammalia", "Order": "Carnivora", "Family": "Felidae"},
    "Chicken": {"Class": "Aves", "Order": "Galliformes", "Family": "Phasianidae"},
    "Duck": {"Class": "Aves", "Order": "Anseriformes", "Family": "Anatidae"},
    "Swan": {"Class": "Aves", "Order": "Anseriformes", "Family": "Anatidae"},
    "Eagle": {"Class": "Aves", "Order": "Accipitriformes", "Family": "Accipitridae"},
    "Falcon": {"Class": "Aves", "Order": "Falconiformes", "Family": "Falconidae"},
    "Parrot": {"Class": "Aves", "Order": "Psittaciformes", "Family": "Psittacidae"},
    "Crow": {"Class": "Aves", "Order": "Passeriformes", "Family": "Corvidae"},
    "Peacock": {"Class": "Aves", "Order": "Galliformes", "Family": "Phasianidae"},
    "Penguin": {"Class": "Aves", "Order": "Sphenisciformes", "Family": "Spheniscidae"},
    "Tortoise": {"Class": "Reptilia", "Order": "Testudines", "Family": "Testudinidae"},
    "Crocodile": {"Class": "Reptilia", "Order": "Crocodylia", "Family": "Crocodylidae"},
    "Alligator": {"Class": "Reptilia", "Order": "Crocodylia", "Family": "Alligatoridae"},
    "Komodo Dragon": {"Class": "Reptilia", "Order": "Squamata", "Family": "Varanidae"},
    "Iguana": {"Class": "Reptilia", "Order": "Squamata", "Family": "Iguanidae"},
    "Gecko": {"Class": "Reptilia", "Order": "Squamata", "Family": "Gekkonidae"},
    "Python": {"Class": "Reptilia", "Order": "Squamata", "Family": "Pythonidae"},
    "Boa Constrictor": {"Class": "Reptilia", "Order": "Squamata", "Family": "Boidae"},
    "Rattlesnake": {"Class": "Reptilia", "Order": "Squamata", "Family": "Viperidae"},
    "Anaconda": {"Class": "Reptilia", "Order": "Squamata", "Family": "Boidae"},
    "Frog": {"Class": "Amphibia", "Order": "Anura", "Family": "Ranidae"},
    "Toad": {"Class": "Amphibia", "Order": "Anura", "Family": "Bufonidae"},
    "Salamander": {"Class": "Amphibia", "Order": "Caudata", "Family": "Salamandridae"},
    "Newt": {"Class": "Amphibia", "Order": "Caudata", "Family": "Salamandridae"},
    "Goldfish": {"Class": "Actinopterygii", "Order": "Cypriniformes", "Family": "Cyprinidae"},
    "Clownfish": {"Class": "Actinopterygii", "Order": "Perciformes", "Family": "Pomacentridae"},
    "Shark": {"Class": "Chondrichthyes", "Order": "Carcharhiniformes", "Family": "Carcharhinidae"},
    "Great White Shark": {"Class": "Chondrichthyes", "Order": "Lamniformes", "Family": "Lamnidae"},
    "Ray": {"Class": "Chondrichthyes", "Order": "Myliobatiformes", "Family": "Myliobatidae"},
    "Octopus": {"Class": "Cephalopoda", "Order": "Octopoda", "Family": "Octopodidae"},
    "Squid": {"Class": "Cephalopoda", "Order": "Teuthida", "Family": "Loliginidae"},
    "Jellyfish": {"Class": "Scyphozoa", "Order": "Semaeostomeae", "Family": "Ulmaridae"},
    "Crab": {"Class": "Malacostraca", "Order": "Decapoda", "Family": "Portunidae"},
    "Lobster": {"Class": "Malacostraca", "Order": "Decapoda", "Family": "Nephropidae"},
    "Shrimp": {"Class": "Malacostraca", "Order": "Decapoda", "Family": "Penaeidae"},
    "Ant": {"Class": "Insecta", "Order": "Hymenoptera", "Family": "Formicidae"},
    "Bee": {"Class": "Insecta", "Order": "Hymenoptera", "Family": "Apidae"},
    "Wasp": {"Class": "Insecta", "Order": "Hymenoptera", "Family": "Vespidae"},
    "Butterfly": {"Class": "Insecta", "Order": "Lepidoptera", "Family": "Nymphalidae"},
    "Moth": {"Class": "Insecta", "Order": "Lepidoptera", "Family": "Noctuidae"},
    "Mosquito": {"Class": "Insecta", "Order": "Diptera", "Family": "Culicidae"},
    "Housefly": {"Class": "Insecta", "Order": "Diptera", "Family": "Muscidae"},
    "Dragonfly": {"Class": "Insecta", "Order": "Odonata", "Family": "Libellulidae"},
    "Grasshopper": {"Class": "Insecta", "Order": "Orthoptera", "Family": "Acrididae"},
    "Cockroach": {"Class": "Insecta", "Order": "Blattodea", "Family": "Blattidae"},
    "Spider": {"Class": "Arachnida", "Order": "Araneae", "Family": "Araneidae"},
    "Scorpion": {"Class": "Arachnida", "Order": "Scorpiones", "Family": "Buthidae"},
    "Earthworm": {"Class": "Clitellata", "Order": "Haplotaxida", "Family": "Lumbricidae"},
    "Snail": {"Class": "Gastropoda", "Order": "Stylommatophora", "Family": "Helicidae"},
    "Slug": {"Class": "Gastropoda", "Order": "Stylommatophora", "Family": "Arionidae"},
    "Starfish": {"Class": "Asteroidea", "Order": "Forcipulatida", "Family": "Asteriidae"},
    "Sea Urchin": {"Class": "Echinoidea", "Order": "Echinoida", "Family": "Echinidae"},
    "Sea Cucumber": {"Class": "Holothuroidea", "Order": "Aspidochirotida", "Family": "Holothuriidae"},
    "Jellyfish": {"Class": "Scyphozoa", "Order": "Semaeostomeae", "Family": "Ulmaridae"},
    "Coral": {"Class": "Anthozoa", "Order": "Scleractinia", "Family": "Acroporidae"},
    "Clam": {"Class": "Bivalvia", "Order": "Venerida", "Family": "Veneridae"},
    "Oyster": {"Class": "Bivalvia", "Order": "Ostreoida", "Family": "Ostreidae"},
    "Squid": {"Class": "Cephalopoda", "Order": "Teuthida", "Family": "Loliginidae"},
    "Octopus": {"Class": "Cephalopoda", "Order": "Octopoda", "Family": "Octopodidae"},
    "Cuttlefish": {"Class": "Cephalopoda", "Order": "Sepiida", "Family": "Sepiidae"},
    "Tuna": {"Class": "Actinopterygii", "Order": "Scombriformes", "Family": "Scombridae"},
    "Salmon": {"Class": "Actinopterygii", "Order": "Salmoniformes", "Family": "Salmonidae"},
    "Carp": {"Class": "Actinopterygii", "Order": "Cypriniformes", "Family": "Cyprinidae"},
    "Eel": {"Class": "Actinopterygii", "Order": "Anguilliformes", "Family": "Anguillidae"},
    "Mackerel": {"Class": "Actinopterygii", "Order": "Scombriformes", "Family": "Scombridae"},
    "Trout": {"Class": "Actinopterygii", "Order": "Salmoniformes", "Family": "Salmonidae"},
    "Anchovy": {"Class": "Actinopterygii", "Order": "Clupeiformes", "Family": "Engraulidae"},
    "Herring": {"Class": "Actinopterygii", "Order": "Clupeiformes", "Family": "Clupeidae"},
    "Swordfish": {"Class": "Actinopterygii", "Order": "Perciformes", "Family": "Xiphiidae"},
    "Pufferfish": {"Class": "Actinopterygii", "Order": "Tetraodontiformes", "Family": "Tetraodontidae"},
    "Seahorse": {"Class": "Actinopterygii", "Order": "Syngnathiformes", "Family": "Syngnathidae"},
    "Lionfish": {"Class": "Actinopterygii", "Order": "Scorpaeniformes", "Family": "Scorpaenidae"},
    "Alligator": {"Class": "Reptilia", "Order": "Crocodilia", "Family": "Alligatoridae"},
    "Crocodile": {"Class": "Reptilia", "Order": "Crocodilia", "Family": "Crocodylidae"},
    "Newt": {"Class": "Amphibia", "Order": "Urodela", "Family": "Salamandridae"},
    "Toad": {"Class": "Amphibia", "Order": "Anura", "Family": "Bufonidae"},
    "Salamander": {"Class": "Amphibia", "Order": "Urodela", "Family": "Salamandridae"},
    "Gila Monster": {"Class": "Reptilia", "Order": "Squamata", "Family": "Helodermatidae"},
    "Iguana": {"Class": "Reptilia", "Order": "Squamata", "Family": "Iguanidae"},
    "Monitor Lizard": {"Class": "Reptilia", "Order": "Squamata", "Family": "Varanidae"},
    "Komodo Dragon": {"Class": "Reptilia", "Order": "Squamata", "Family": "Varanidae"},
    "Chameleon": {"Class": "Reptilia", "Order": "Squamata", "Family": "Chamaeleonidae"},
    "Gecko": {"Class": "Reptilia", "Order": "Squamata", "Family": "Gekkonidae"},
    "Horned Lizard": {"Class": "Reptilia", "Order": "Squamata", "Family": "Phrynosomatidae"},
    "Sea Turtle": {"Class": "Reptilia", "Order": "Testudines", "Family": "Cheloniidae"},
    "Snapping Turtle": {"Class": "Reptilia", "Order": "Testudines", "Family": "Chelydridae"},
    "Box Turtle": {"Class": "Reptilia", "Order": "Testudines", "Family": "Emydidae"},
    "Rattlesnake": {"Class": "Reptilia", "Order": "Squamata", "Family": "Viperidae"},
    "Boa Constrictor": {"Class": "Reptilia", "Order": "Squamata", "Family": "Boidae"},
    "Python": {"Class": "Reptilia", "Order": "Squamata", "Family": "Pythonidae"},
    "Anaconda": {"Class": "Reptilia", "Order": "Squamata", "Family": "Boidae"},
    "Green Tree Python": {"Class": "Reptilia", "Order": "Squamata", "Family": "Pythonidae"},
    "Peacock": {"Class": "Aves", "Order": "Galliformes", "Family": "Phasianidae"},
    "Penguin": {"Class": "Aves", "Order": "Sphenisciformes", "Family": "Spheniscidae"},
    "Hawk": {"Class": "Aves", "Order": "Accipitriformes", "Family": "Accipitridae"},
    "Eagle": {"Class": "Aves", "Order": "Accipitriformes", "Family": "Accipitridae"},
    "Owl": {"Class": "Aves", "Order": "Strigiformes", "Family": "Strigidae"},
    "Woodpecker": {"Class": "Aves", "Order": "Piciformes", "Family": "Picidae"},
    "Parrot": {"Class": "Aves", "Order": "Psittaciformes", "Family": "Psittacidae"},
    "Swan": {"Class": "Aves", "Order": "Anseriformes", "Family": "Anatidae"},
    "Flamingo": {"Class": "Aves", "Order": "Phoenicopteriformes", "Family": "Phoenicopteridae"},
    "Duck": {"Class": "Aves", "Order": "Anseriformes", "Family": "Anatidae"},
    "Goose": {"Class": "Aves", "Order": "Anseriformes", "Family": "Anatidae"},
    "Rooster": {"Class": "Aves", "Order": "Galliformes", "Family": "Phasianidae"},
    "Turkey": {"Class": "Aves", "Order": "Galliformes", "Family": "Phasianidae"},
    "Canary": {"Class": "Aves", "Order": "Passeriformes", "Family": "Fringillidae"},
    "Pigeon": {"Class": "Aves", "Order": "Columbiformes", "Family": "Columbidae"},
    "Sparrow": {"Class": "Aves", "Order": "Passeriformes", "Family": "Passeridae"},
    "Seagull": {"Class": "Aves", "Order": "Charadriiformes", "Family": "Laridae"},
    "Falcon": {"Class": "Aves", "Order": "Falconiformes", "Family": "Falconidae"},
    "Robin": {"Class": "Aves", "Order": "Passeriformes", "Family": "Turdidae"},
    "Crow": {"Class": "Aves", "Order": "Passeriformes", "Family": "Corvidae"},
    "Leopard": {"Class": "Mammalia", "Order": "Carnivora", "Family": "Felidae"},
    "Cheetah": {"Class": "Mammalia", "Order": "Carnivora", "Family": "Felidae"},
    "Jaguar": {"Class": "Mammalia", "Order": "Carnivora", "Family": "Felidae"},
    "Hyena": {"Class": "Mammalia", "Order": "Carnivora", "Family": "Hyaenidae"},
    "Panther": {"Class": "Mammalia", "Order": "Carnivora", "Family": "Felidae"},
    "Wild Boar": {"Class": "Mammalia", "Order": "Artiodactyla", "Family": "Suidae"},
    "Bison": {"Class": "Mammalia", "Order": "Artiodactyla", "Family": "Bovidae"},
    "Wildebeest": {"Class": "Mammalia", "Order": "Artiodactyla", "Family": "Bovidae"},
    "Moose": {"Class": "Mammalia", "Order": "Artiodactyla", "Family": "Cervidae"},
    "Reindeer": {"Class": "Mammalia", "Order": "Artiodactyla", "Family": "Cervidae"},
    "Gazelle": {"Class": "Mammalia", "Order": "Artiodactyla", "Family": "Bovidae"},
    "Antelope": {"Class": "Mammalia", "Order": "Artiodactyla", "Family": "Bovidae"},
    "Tapir": {"Class": "Mammalia", "Order": "Perissodactyla", "Family": "Tapiridae"},
    "Sloth": {"Class": "Mammalia", "Order": "Pilosa", "Family": "Bradypodidae"},
    "Armadillo": {"Class": "Mammalia", "Order": "Cingulata", "Family": "Dasypodidae"},
    "Okapi": {"Class": "Mammalia", "Order": "Artiodactyla", "Family": "Giraffidae"},
    "Gaur": {"Class": "Mammalia", "Order": "Artiodactyla", "Family": "Bovidae"},
    "Takin": {"Class": "Mammalia", "Order": "Artiodactyla", "Family": "Bovidae"},
    "Capybara": {"Class": "Mammalia", "Order": "Rodentia", "Family": "Caviidae"},
    "Aardvark": {"Class": "Mammalia", "Order": "Tubulidentata", "Family": "Orycteropodidae"},
}

def classify(name):
    if name in HARDCODE:
        return HARDCODE[name]
    qid = resolve_qid(name)
    if not qid:
        return None
    result = {}
    result.update(sparql_rank(qid))
    if len(result) < 3:
        result = rest_taxonomy(qid, result)
    return result if len(result) == 3 else None



if st.button("Generate Information") and animal_input:
    name = animal_input.strip()
    summary = get_wikipedia_summary(name)
    tax = classify(name)

    st.subheader("📖 Summary")
    st.write(summary or "No summary available.")

    st.subheader("🧬 Scientific Classification")
    if tax:
        st.markdown(f"- **Class:** {tax['Class']}")
        st.markdown(f"- **Order:** {tax['Order']}")
        st.markdown(f"- **Family:** {tax['Family']}")
    else:
        st.error("Complete scientific classification could not be found automatically.")
