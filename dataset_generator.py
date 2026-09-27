"""
dataset_generator.py
Generates a curated, balanced dataset of Human-written and AI-generated texts
across diverse domains (Technology, Science, Environment, Education, Society, Literature).
"""

import os
import pandas as pd

HUMAN_TEXTS = [
    # Technology & AI
    "Honestly, I spent three hours yesterday just trying to get this printer to connect to my Wi-Fi. Every time tech promises to make our lives seamless, we end up staring at blinking lights wondering why Bluetooth suddenly decided to give up on us.",
    "My grandmother called me in a panic because her iPad screen rotated and she thought the entire device broke. It took twenty minutes of explaining before she realized she could just twist it back around. Tech literacy is wild.",
    "I'm not completely convinced that automated driving is ready for Indian roads or rush hour anywhere. Have you seen how people cut across three lanes without warning? An algorithm would freeze up from pure panic in five seconds.",
    "We keep building smart gadgets for things nobody asked for. A smart toaster that sends push notifications to my phone when bread is browned? Why? Just look inside the slot with your own eyes!",
    "Coding at 2 AM with a lukewarm mug of instant coffee hits different. You fix one missing semicolon and suddenly three new bugs emerge out of nowhere like mythical hydra heads.",
    "I deleted Instagram last month, and to be honest, my attention span improved almost immediately. I don't feel that nagging urge to document my morning oatmeal anymore.",
    "People worry about superintelligent robots taking over the world, but yesterday my phone autocorrected 'definitely' to 'deflated' four times in a row. We are safe for a while.",
    "The open-source community is truly one of humanity's finest achievements. Complete strangers across continents collaborating for free just to build Linux or Blender because they genuinely love building things.",
    "Every software update seems to move things around just for the sake of making it look different. Why bury the audio settings three menus deeper than they were last week?",
    "I tried building my own mechanical keyboard last summer. Lubing every individual switch was painfully tedious, but the satisfying clack sound made every cramped finger worth it.",

    # Environment & Climate
    "Walking through the woods near my childhood home, I noticed the creek where we used to catch tadpoles has completely dried into a muddy trench. It's heartbreaking to see changes happen right before your eyes within a single decade.",
    "Recycling plastic feels like a personal guilt trip pushed on consumers while massive manufacturing corporations dump tons of untreated runoff into rivers every single hour.",
    "Solar panels on our terrace finally got hooked up to the local grid this Tuesday! Watching the electric meter spin backward in the blazing afternoon heat was oddly exhilarating.",
    "The monsoon this year felt completely erratic. Usually, June brings steady downpours, but this time we got three weeks of scorching heat followed by two days of flash floods that submerged the ground floor parking.",
    "City parks are not just aesthetic decorations; they are literal lifelines for urban sanity. If I can't sit under a neem or banyan tree after eight hours of Excel spreadsheets, my brain fries.",
    "Composting kitchen scraps seemed intimidating at first, but once you get the dry leaves to wet peelings ratio right, you end up with rich black soil that makes tomato plants go crazy.",
    "I hate how fast fashion dominates every shopping mall now. Shirts that fall apart after two washes and shed polyester microfibers into the water supply are sold as trendy disposable commodities.",
    "Electric vehicles are a step forward, sure, but what happens to the massive lithium batteries fifteen years down the road? We need circular recycling, not just trading one extraction mess for another.",
    "There's something deeply humbling about hiking up a ridge at sunrise and watching fog roll through the valley below. You realize just how tiny human concerns really are.",
    "I planted three saplings along our boundary wall five years ago. Today they are taller than our second-floor balcony, filled with sparrows every evening.",

    # Education & College Life
    "Cramming four chapters of operational research the night before the semester exam is a rite of passage no engineering student escapes. You swear you'll study early next semester, but you never do.",
    "The best professors aren't the ones who know the most formulas; they are the ones who can explain why a formula actually matters without making you feel stupid for asking a basic question.",
    "Group projects are an unpaid lesson in human psychology. One person does 90 percent of the research, one formats the slides, and two show up on presentation day smiling like they co-authored the thesis.",
    "I remember sitting in the library basement surrounded by photocopied lecture notes, trying to decipher handwriting from three weeks ago. College pre-exam stress is a distinct brand of chaos.",
    "Online lectures during the pandemic were surreal. Half the class had their cameras turned off, somebody's dog was barking in the background, and the professor spoke while muted for fifteen whole minutes.",
    "Internship hunting in your third year feels like an endless loop of uploading your resume into portals that ask you to manually re-type every single section of the resume you just uploaded.",
    "Hands-on lab experiments teach you twice as much as three textbook derivations. When a circuit actually smokes because you wired a resistor wrong, you never forget Ohm's law again.",
    "Standardized entrance exams measure speed and test-taking tricks far more than actual curiosity or deep conceptual mastery. It burns kids out before they even choose a career.",
    "Sharing cold samosas and chai with batchmates in the canteen after surviving a brutal viva exam is where real friendships are forged.",
    "I changed my major twice before figuring out what I actually wanted to work on. Society rushes 18-year-olds into lifetime decisions way too early.",

    # Health, Food & Everyday Life
    "Making sourdough bread taught me patience like nothing else. You feed a flour-and-water paste for a week, wait twelve hours for dough proofing, and if the temperature drops two degrees, your loaf turns into a doorstop.",
    "My dad refuses to follow GPS instructions. He insists on asking the roadside tea stall vendor for directions because 'a local person knows shortcuts no satellite can detect.' And half the time, he's actually right.",
    "There's nothing quite like street-side pani puri on a humid evening. The crisp puri bursting with spicy tangy mint water instantly wipes out a stressful workday.",
    "Going to the gym consistently is 80 percent mental resistance. Once you put your running shoes on and step through the door, the workout itself is never as bad as lying on the couch dreading it.",
    "Home-cooked dal tadka with freshly steamed basmati rice and a dollop of ghee beats any five-star restaurant meal on a rainy Sunday afternoon.",
    "Sleep deprivation is insidious. You convince yourself you function fine on five hours, until you try to remember where you parked your scooter or put your keys in the refrigerator.",
    "Learning to cook as an adult is thrilling because you realize restaurant magic is mostly just generous amounts of butter, garlic, and high heat.",
    "I tried that trending green smoothie recipe with kale and ginger. It tasted like lawn clippings soaked in lemon juice. Life is too short to drink grass pulp for breakfast.",
    "Morning routines with twenty-five steps recommended by productivity influencers are absurd. Get out of bed, wash your face, drink some water, and get to work.",
    "Reading an actual paper book with physical pages before bed beats scrolling through endless blue-light feeds every single night.",

    # Literature, History & Arts
    "Kafka had it right: bureaucratic systems aren't intentionally evil; they are just cold labyrinths of procedural indifference where individual humanity gets stamped out by carbon copies.",
    "Visiting ancient stone temples carved out of single granite boulders makes modern steel architecture look oddly transient. How did artisans without lasers carve such perfect symmetries a thousand years ago?",
    "Good poetry doesn't decorate language; it strips away pretension until only raw emotional truth is left bare on the page.",
    "Music has this uncanny power of emotional time travel. A five-second melody from a song you listened to in 2018 instantly transports you back to that specific bus seat and sunset.",
    "History is written by survivors, but folk ballads preserve what ordinary villagers actually suffered while kings fought over boundary markers.",
    "Black and white cinema holds a dramatic weight that saturated 4K CGI often misses. When light and shadow do all the storytelling, every facial flicker matters.",
    "I tried writing a short novel during quarantine. It took fifty pages before I realized my protagonist had no goal, no obstacle, and was basically just me whining on paper.",
    "Languages carry unique worldviews. Untranslatable words like 'saudade' or 'jugaad' prove that human culture can't be compressed into standardized dictionary definitions.",
    "A museum without proper context is just a warehouse of old relics. But when a good curator tells the human story behind a rusty bronze coin, history suddenly comes alive.",
    "Painting with watercolors is brutal because mistakes can't be painted over with white acrylic. You either embrace the accidental bleed or start from scratch on fresh paper."
]

AI_TEXTS = [
    # Technology & AI
    "Artificial intelligence has emerged as a transformative paradigm across modern industries. By leveraging complex statistical patterns and machine learning algorithms, modern systems can process unstructured data with remarkable efficiency. Furthermore, continuous advances in computational hardware have accelerated deployment cycles.",
    "Cloud computing architecture facilitates seamless scalability for enterprise applications. It is important to recognize that virtualization optimizes resource allocation while mitigating localized hardware dependencies. In addition, distributed databases ensure high availability across redundant availability zones.",
    "The integration of Internet of Things (IoT) sensors within smart infrastructure enables automated monitoring and predictive maintenance. Consequently, operational bottlenecks can be proactively identified and resolved before causing systemic disruptions.",
    "Cybersecurity protocols must continuously evolve to address emerging cryptographic challenges. Multi-factor authentication, endpoint encryption, and automated intrusion detection systems play a crucial role in safeguarding sensitive user information.",
    "Natural language processing has witnessed substantial milestones with the advent of transformer architectures. These architectures utilize self-attention mechanisms to weigh contextual token relationships across bidirectional contexts, yielding coherent textual synthesis.",
    "Autonomous vehicles utilize a combination of computer vision, LiDAR, and radar telemetry to navigate dynamic traffic environments. It is worth noting that sensor fusion algorithms are essential for synthesizing disparate sensory inputs into accurate spatial maps.",
    "Blockchain technology offers a decentralized ledger system characterized by immutability and cryptographic verification. By eliminating centralized intermediaries, smart contracts facilitate transparent transactions across peer-to-peer networks.",
    "Edge computing processes computational tasks locally rather than relying exclusively on distant data centers. This paradigm significantly diminishes latency and bandwidth consumption, making it particularly advantageous for real-time applications.",
    "Software containerization via tools like Docker and Kubernetes streamlines application deployment across heterogeneous environments. This methodology fosters modular development practices and facilitates robust microservices architecture.",
    "Quantum computing represents a fundamental departure from classical binary computation. Utilizing qubits that exist in superposition states, quantum processors possess the potential to solve intractable optimization problems efficiently.",

    # Environment & Climate
    "Climate change poses significant ecological and economic risks to global sustainability. Rising greenhouse gas emissions have catalyzed global temperature anomalies, leading to accelerated glacial retreat and frequent meteorological extremes. Addressing this imperative necessitates coordinated multilateral interventions.",
    "Renewable energy systems, including photovoltaic solar arrays and wind turbines, are pivotal in transitioning toward carbon-neutral infrastructure. Moreover, investment in battery energy storage systems addresses the intermittency challenges inherent to renewable generation.",
    "Biodiversity conservation is essential for maintaining ecosystem resilience and supporting ecological services. Habitat fragmentation, deforestation, and anthropogenic pollution undermine critical ecological equilibria, necessitating rigorous preservation policies.",
    "Urban forestry initiatives provide significant microclimate moderation and air quality improvements in dense metropolitan regions. Furthermore, permeable pavements and green roofs mitigate urban heat island effects while enhancing stormwater management.",
    "The circular economy model seeks to decouple economic growth from finite resource extraction. By prioritizing durable product design, modular repairability, and comprehensive recycling streams, industrial systems can dramatically minimize aggregate waste generation.",
    "Ocean acidification, driven by elevated atmospheric carbon dioxide absorption, poses severe threats to marine calcifying organisms and coral reef ecosystems. Consequently, safeguarding marine habitats requires targeted conservation efforts alongside global emission abatement.",
    "Sustainable agricultural methodologies, such as precision irrigation and regenerative soil practices, enhance agricultural yields while conserving vital freshwater reserves. These techniques ensure food security in the face of shifting climatic patterns.",
    "Deforestation in tropical rainforests diminishes global carbon sequestration capacity and accelerates species extinction rates. Implementing satellite-based canopy monitoring and enforcing sustainable land-use regulations are indispensable components of international conservation strategies.",
    "Atmospheric particulate matter, specifically PM2.5, contributes significantly to respiratory and cardiovascular mortality rates in industrial corridors. Stricter emissions compliance and low-emission transit alternatives are vital for public health preservation.",
    "Water resource management requires balanced allocations between agricultural demands, industrial consumption, and domestic sanitation. Automated telemetry systems assist municipal planners in optimizing water delivery networks efficiently.",

    # Education & Learning
    "Educational pedagogy has transitioned toward learner-centric paradigms facilitated by educational technology platforms. Interactive modules and adaptive learning algorithms customize curriculum delivery based on individual student comprehension trajectories.",
    "Collaborative learning environments foster critical thinking and interpersonal communication skills among academic cohorts. Group discussions and peer review processes encourage students to evaluate multifaceted perspectives and construct well-reasoned arguments.",
    "Assessment methodologies are increasingly incorporating formative evaluation frameworks alongside traditional summative examinations. This balanced approach provides continuous feedback mechanisms that support incremental learning outcomes.",
    "Experiential learning bridges theoretical knowledge and practical execution by engaging learners in authentic problem-solving scenarios. Laboratory simulations and field internships provide invaluable context for abstract academic principles.",
    "Digital literacy is an indispensable competency within twenty-first-century educational curricula. Equipping learners with information verification skills and algorithmic awareness enables critical navigation of information ecosystems.",
    "Interdisciplinary research initiatives encourage the convergence of scientific inquiry, technological innovation, and humanities scholarship. Such integrative frameworks are essential for addressing complex global challenges effectively.",
    "Universal design in learning emphasizes inclusive instructional strategies that accommodate diverse cognitive and sensory learning requirements. Accessible instructional media ensures equitable educational access for all learners.",
    "Vocational education programs provide targeted technical competencies aligned with industrial requirements. Establishing robust partnerships between vocational institutions and industry stakeholders enhances graduate employability and economic productivity.",
    "Standardized evaluation mechanisms provide benchmarking metrics across educational jurisdictions. However, educational researchers emphasize that evaluations must be balanced with holistic qualitative measures of student development.",
    "Lifelong learning has become a prerequisite for career progression within rapidly evolving knowledge economies. Continuing education modules and modular credentials enable professionals to acquire emergent skills throughout their careers.",

    # Health & Lifestyle
    "Regular aerobic exercise is documented to confer comprehensive cardiovascular and cognitive benefits. Physical exertion stimulates neurotrophic factor release, thereby supporting neuroplasticity and emotional well-being across demographic cohorts.",
    "Nutritional science emphasizes the consumption of balanced macronutrient profiles alongside diverse micronutrients and dietary fiber. Minimizing ultra-processed foods containing high sodium and refined sugars reduces long-term risks of metabolic disorders.",
    "Circadian rhythm synchronization plays a fundamental role in maintaining hormonal regulation, immune function, and cellular repair processes. Maintaining consistent sleep hygiene schedules supports optimal cognitive performance.",
    "Preventative healthcare strategies focus on early diagnostic screening and lifestyle interventions to attenuate chronic illness incidence. Proactive wellness management represents a cost-effective alternative to reactive acute medical interventions.",
    "Mental health awareness initiatives emphasize destigmatizing psychiatric conditions and expanding access to evidence-based psychological support. Integrative treatment modalities combining behavioral therapy and lifestyle adjustments yield favorable clinical outcomes.",
    "Hydration is critical for cellular metabolic processes, thermoregulation, and joint lubrication. Adequate daily fluid intake ensures the preservation of cognitive alertness and renal physiological function.",
    "Workplace ergonomics promotes musculoskeletal health by optimizing posture, workstation arrangement, and task distribution. Implementing ergonomic standards substantially decreases occupational strain injuries and improves workplace productivity.",
    "Microbiome research underscores the intricate bidirectional communication pathway between gut flora and neurological systems. Consuming prebiotic fibers and fermented foods supports microbial diversity, conferring systemic physiological benefits.",
    "Chronic physiological stress elevates systemic cortisol levels, potentially impairing immune function and cardiovascular health. Incorporating structured relaxation techniques, such as mindfulness meditation, assists in mitigating autonomic nervous system arousal.",
    "Immunization programs constitute one of the most cost-effective public health interventions in modern medicine. Herd immunity thresholds protect vulnerable populations from infectious pathologies through systematic vaccination coverage.",

    # Literature & History
    "Literary narratives reflect the socio-political dynamics and philosophical inquiries of their respective historical epochs. Authors employ figurative language, thematic allegory, and character arcs to interrogate cultural values and human motivations.",
    "The Renaissance epoch catalyzed profound shifts in European intellectual and artistic inquiry. By revitalizing classical humanistic traditions, scholars and artists fostered empirical observation that laid the groundwork for modern scientific thought.",
    "Historical historiography requires critical evaluation of primary documentation, archaeological artifacts, and contextual perspectives. Discerning inherent biases in archival records is essential for constructing objective historical narratives.",
    "Architectural heritage serves as tangible evidence of civilizational achievements, aesthetic philosophies, and engineering prowess. Conserving historical monuments ensures cultural continuity and enriches our understanding of ancestral societies.",
    "Philosophical discourse concerning ethical theory distinguishes between deontological imperatives, consequentialist outcomes, and virtue ethics. Analyzing moral dilemmas through these frameworks clarifies fundamental principles governing societal justice.",
    "Poetic structures, ranging from sonnets to blank verse, exploit metrical rhythms and phonological patterns to evoke profound cognitive and affective responses. Structural constraints often amplify thematic expression.",
    "The Industrial Revolution fundamentally altered global economic landscapes, labor dynamics, and demographic distributions. While rapid technological mechanization expanded production efficiency, it simultaneously prompted complex labor welfare questions.",
    "Linguistic typology studies structural and grammatical variation across world languages. Comparative linguistics reveals systemic morphological evolutions and historical migration pathways across language families.",
    "Museum curation bridges academic research and public engagement by presenting curated collections within coherent interpretive frameworks. Curatorial strategies stimulate critical dialogue regarding cultural representation.",
    "Dramatic theater provides an immersive medium for exploring existential dilemmas, interpersonal conflict, and societal critique. The synthesis of stagecraft, vocal delivery, and narrative drama produces cathartic audience experiences."
]


def expand_dataset():
    """
    Expands the seed texts into a rich, robust dataset with variations,
    combinations, and contextual paragraphs, creating 200 Human and 200 AI samples.
    """
    data = []

    # 1. Base human samples
    for text in HUMAN_TEXTS:
        data.append({"text": text.strip(), "label": 0, "source": "Human"})

    # 2. Base AI samples
    for text in AI_TEXTS:
        data.append({"text": text.strip(), "label": 1, "source": "AI"})

    # 3. Create realistic compound human paragraphs (multi-sentence natural writing)
    for i in range(len(HUMAN_TEXTS)):
        # Pair adjacent human texts with personal transitions
        t1 = HUMAN_TEXTS[i]
        t2 = HUMAN_TEXTS[(i + 3) % len(HUMAN_TEXTS)]
        compound_human_1 = f"{t1} Honestly, thinking about it now, that's just how life goes. {t2}"
        compound_human_2 = f"I was talking to a friend about this exact thing recently. {t2} To be frank, it really makes you reconsider your habits. {t1}"
        compound_human_3 = f"Here is my hot take on this: {t1} And don't get me started on the rest of it! {t2}"
        data.append({"text": compound_human_1, "label": 0, "source": "Human"})
        data.append({"text": compound_human_2, "label": 0, "source": "Human"})
        data.append({"text": compound_human_3, "label": 0, "source": "Human"})

    # 4. Create realistic compound AI paragraphs (multi-sentence structured synthesis)
    for i in range(len(AI_TEXTS)):
        # Pair adjacent AI texts with formal academic transitions
        t1 = AI_TEXTS[i]
        t2 = AI_TEXTS[(i + 3) % len(AI_TEXTS)]
        compound_ai_1 = f"{t1} In addition to these considerations, it is crucial to analyze the broader implications. {t2}"
        compound_ai_2 = f"When evaluating modern paradigms, several interconnected factors emerge. {t1} Consequently, experts emphasize the need for balanced strategies. {t2}"
        compound_ai_3 = f"Furthermore, systematic investigations reveal substantial correlations. {t2} Ultimately, achieving sustainable outcomes requires ongoing coordination. {t1}"
        data.append({"text": compound_ai_1, "label": 1, "source": "AI"})
        data.append({"text": compound_ai_2, "label": 1, "source": "AI"})
        data.append({"text": compound_ai_3, "label": 1, "source": "AI"})

    df = pd.DataFrame(data)
    # Shuffle dataset deterministically
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    dataset = expand_dataset()
    output_path = os.path.join("data", "ai_vs_human_dataset.csv")
    dataset.to_csv(output_path, index=False)
    print(f"Dataset generated successfully at {output_path}!")
    print(f"Total samples: {len(dataset)}")
    print(f"Class distribution:\n{dataset['source'].value_counts()}")
