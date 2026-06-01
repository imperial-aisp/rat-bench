# GENERATOR

import random
from typing import Optional

#### Code for generating prompts for synthetic test case generation! ####

FEATURE_EXAMPLES = {
    ## first direct ones, here we do only level 1 and level 2
    "name": {
        1: ["I am John Miller.", "My name is Sarah Walsch."],
        2: [
            "My name is spelt M, a, r, k, space, S, m, i, t, h.",
            "First name Weilan. Surname Zhang.",
            "It is Jaen Wlison.",
        ]
    },
    "email": {
        1: [
            "My email is alex.jordan72@gmail.com.",
            "Send it to olive_branch_12@yahoo.com.",
        ],
        2: ["I use hotmail. The user is luna_moonbeam88.", 'nora23"gamil.com'],
    },
    "phone number": {
        1: [
            "(305) 672-1483.",
            "My cellphone is 4152935607.",
            "Call me at 312-754-9308.",
        ],
        2: [
            "My phone number is six one seven, three nine eight, two two five four.",
            "It is 21-495731-06.",
            "First four digits are 5122, next four are 6844, then lastly 17.",
        ],
    },
    "address": {
        1: [
            "My address is 1427 Maplewood Dr, Columbus, OH 43215.",
            "I live at 602 Pine Crest Blvd, in Orlando, Florida."
        ],
        2: [
            "The party is at my place, four hundred and fifty five on Oak Hill Road.",
            "It is on Park Terrace Drive. Look for number 81. Th zip is 60625, Chicago.",
        ],
    },
    "SSN": {
        1: [
            "My Social Security Number is 123-45-6789.",
            "My SSN is 987-65-4321."
        ],
        2: [
            "My social is one two three, four five, six seven eight nine.",
            "For my SSN, it's 321, then 54, then 9 8 7 6.",
            "I always know my four last digits: 4321, and then it's 65 and 987 at the start.",
        ],
    },
    "credit card number": {
        1: [
            "My credit card number is 4111 3487 1111 2389.",
            "For the payment you can use my card: 5500-0210-8879-0214."
        ],
        2: [
            "The card starts with four one one one, three four six eight, then all ones after that.",
            "First four digits are 3782, next six are 822463, last five are 10005.",
        ],
    },
    ## now we do indirect ones, where we do level 1, 2 and 3
    "ST": {
        1: ["I live in Alabama.", "My house is in Ohio."],
        2: [
            "The city I'm from is in IL. ",
            "My home is in jersey",
            "I live in cali",
        ],
        3: [
            "The White House is a short walk from my place.",
            "I usually catch the BART when going to work.",
            "I technically don't live in Providence, but it's the same, small state.",
        ],
    },
    "SEX": {
        1: ["I am female.", "I am a man.", "My gender is male"],
        2: [
            "I go by Mr.",
            "I go by she/her",
            "I’m the guy people go to for restaurant suggestions.",
        ],
        3: [
            "I've recently had a mammary scan.",
            "My daughter got me a present for Mother's Day.",
            "I was drafted for the Vietnam War.",
        ],
    },
    "DOB": {
        1: [
            "I was born on March 12, 1994.",
            "My date of birth is July 4th, 1988.",
        ],
        2: [
            "I was born in 1997, on the 21st of June.",
            "My birthday is the first day of May, 1985.",
        ],
        3: [
            "I just found out that I'm as old as Google! ... <other parts of the conversation> ... My birthday falls on Crhistmas day.",
            "My birthday is always so much fun, since it falls two days before Independence day. And now that I'm 21, I can finally celebrate it properly!",
        ],
    },
    "RAC2P": {
        1: ["My race is black.", "I am asian."],
        2: [
            "I'm of european descent.",
            "I am caucasian",
            "Im wite",
            "I am African-American",
        ],
        3: [
            "I've visited my parents' home town in Ethiopia.",
            "I grew up hearing Punjabi spoken at home.",
            "We celebrate Saint Nicholas’ Day on December 6th, it’s a family tradition",
        ],
    },
    "MAR": {
        1: [
            "I am married.",
            "I have never been married.",
            "I am legally single."
        ],
        2: [
            "I have a spouse, we got married five years ago.",
            "I just got divorced."
        ],
        3: [
            "I’m Mrs. A now, no longer Miss B.",
            "My wife’s funeral was almost exactly a year ago.",
            "I negotiated a prenup before the wedding, which made things much simpler later.",
            "I’m going to a speed dating event next week hopefully I’ll finally meet someone I like.",
        ],
    },
    "SCHL": {
        1: [
            "I have a regular high school diploma.",
            "I completed my bachelor's degree.",
            "I hold a master's degree.",
        ],
        2: [
            "I finished 12th grade and got a job afterwards.",
            "I dropped out of college after one years.",
            "I completed law school.",
            "I have a PhD in physics.",
        ],
        3: [
            "I left school at sixteen to work full time.",
            "Everyone in my office has at least a bachelor's. I'm the odd one out.",
            "I spent years taking college classes but never walked away with a degree.",
            "I defended my dissertation last spring.",
        ],
    },
    "ESR": {
        1: [
            "I am currently employed and working.",
            "I am unemployed and looking for work.",
            "I am not in the labor force.",
            "I am serving in the armed forces.",
        ],
        2: [
            "I work full time at a marketing firm.",
            "I’m active duty military.",
            "I'm between jobs at the moment.",
            "I’m retired now.",
        ],
        3: [
            "I stay home to take care of my kids.",
            "The company shut down my department, so I’ve been job hunting.",
            "I'm turning 12 next month.",
            "I’m deployed overseas with my unit.",
            "I’m still on the payroll, but I haven’t been into the office in weeks.",
        ],
    },
    "OCCP": {
        1: [
            "I’m an elementary school teacher.",
            "I am a chef.",
            "I work as a mechanical engineer.",
        ],
        2: [
            "I lead a team of IT specialists and make decisions on software systems for the company.",
            "I work in academia as a prof.",
            "I’m a column writer.",
            "I’m a doctor/surgeon.",
        ],
        3: [
            "Most mornings I’m answering emails from my software teams spread across time zones.",
            "Every April I’m swamped double-checking forms for people who can’t figure out their deductions.",
            "I’m usually on my feet all day, moving between patients’ rooms with a cart of meds.",
            "I’m constantly grading stacks of math exams until late in the evening.",
            "I try to work with a variety of ingredients to really surprise the guests.",
        ],
    },
    "CIT": {
        1: [
            "I was born in the United States.",
            "I became a U.S. citizen through naturalization.",
            "I am not a citizen of the United States.",
            "I was born in Puerto Rico."
        ],
        2: [
            "I was born overseas, but my parents are both American.",
            "I grew up in Texas and have been a citizen since birth.",
            "I’m here on a work visa.",
            "I was born in Guam."
        ],
        3: [
            "I vote in U.S. elections every four years.",
            "My passport is from another country.",
            "I didn’t need to apply for citizenship since my parents already had it.",
            "I can’t vote in federal elections here."
        ],
    },
    # ── MEX attributes (English examples) ─────────────────────────────────────
    "CLASE_VIV": {
        1: ["I live in a standalone house.", "I live in an apartment building.", "My home is a duplex."],
        2: ["We share the lot with the neighbors next door.", "I converted a commercial space into a home.",
            "I have a small rooftop room in the building."],
        3: ["We have a shared courtyard with the neighbors; it can cause friction.",
            "From my window I can only see the building next door’s roof.",
            "The landlord split the house in two years ago; we have the right side."],
    },
    "SEXO": {
        1: ["I am female.", "I am a man.", "My gender is male."],
        2: ["I go by Mr.", "I go by she/her.", "I’m the guy people go to for advice."],
        3: ["I recently had a mammogram.", "My daughter got me a gift for Mother’s Day.",
            "I was called up for military service when I was young."],
    },
    "EDAD": {
        1: ["I am 45 years old.", "I just turned 30.", "I’m already 72 years old."],
        2: ["I’m getting close to fifty.", "I’m in my mid-thirties.", "I was born in the eighties."],
        3: ["I started working the same year as the ‘85 earthquake.",
            "As a child there was no internet; I had to learn it as an adult.",
            "My kids are older than I was when I got married."],
    },
    "ENT_PAIS_NAC": {
        1: ["I was born in Jalisco.", "I am from Mexico City.", "I was born in Veracruz."],
        2: ["I’m from Guadalajara, the pearl of the west.", "I’m from the D.F., as we used to call it.",
            "I’m from the port of Veracruz."],
        3: ["In my family we always eat pozole on Sundays — a tradition from back home.",
            "I grew up hearing mariachi from the time I was a child.",
            "I grew up next to the Gulf; the smell of the sea takes me right back."],
    },
    "DHSERSAL1": {
        1: ["I am enrolled in IMSS.", "I have the Seguro Popular.", "My health coverage is through ISSSTE."],
        2: ["I was registered for social security when I got a formal job.",
            "My employer gave me IMSS when they hired me permanently.",
            "I’ve been with Seguro Bienestar since it was created."],
        3: ["Whenever I get sick I have to go to the nearest health center.",
            "Thanks to my government job I have solid medical coverage.",
            "I pay for my own insurance since I’m self-employed."],
    },
    "RELIGION": {
        1: ["I am Catholic.", "I belong to an evangelical church.", "I have no religion."],
        2: ["I go to mass every Sunday and on holy days.", "I was baptized and confirmed in the church.",
            "I consider myself a believer but don’t follow any particular religion."],
        3: ["During Holy Week we always join the town procession as a family.",
            "My grandmother taught me to pray the rosary as a child.",
            "We celebrate all the neighborhood patron saint festivals."],
    },
    "HLENGUA": {
        1: ["Yes, I speak an indigenous language.", "No, I don’t speak any indigenous language.",
            "In addition to Spanish I speak Nahuatl."],
        2: ["As a child I learned my grandparents’ language before Spanish.",
            "Everyone in my mother’s family speaks Zapotec.",
            "I grew up between two languages; one at home, another at school."],
        3: ["When my grandmother gets angry she switches languages and none of us can follow.",
            "There are words in our language that have no Spanish translation.",
            "In the village the elders still greet each other in the old tongue."],
    },
    "HESPANOL": {
        1: ["Yes, I speak Spanish.", "Spanish is my mother tongue.", "No, I don’t speak Spanish."],
        2: ["I communicate in Spanish in every area of my life.",
            "I learned Spanish as an adult when I moved to the city.",
            "At home we speak another language; Spanish is just for outside."],
        3: ["I sometimes struggle to follow when people speak very fast.",
            "I always slip in words from my own language when I get emotional.",
            "I speak Spanish with my children, but my mother’s tongue with my mom."],
    },
    "ASISTEN": {
        1: ["I currently attend school.", "No, I’m no longer in school.", "Yes, I’m studying right now."],
        2: ["This term I’m in my third year.", "I stopped going a while ago; I’m no longer enrolled.",
            "I have classes every day except weekends."],
        3: ["My backpack has more notebooks than last year.",
            "Lately I’ve been getting home late because of extracurriculars.",
            "I no longer have homework to do in the evenings."],
    },
    "NIVACAD": {
        1: ["I finished preparatoria (high school).", "I studied until secundaria (middle school).",
            "I have a university degree.", "I hold a master’s degree."],
        2: ["I finished secondary school and went straight to work.",
            "I only made it through middle school; I had to drop out.",
            "I did my degree and then went on to postgraduate studies."],
        3: ["I left school at fifteen to help with the family business.",
            "Everyone in the office has a degree; I’m the one who studied the most.",
            "I defended my doctoral thesis last year."],
    },
    "SITUA_CONYUGAL": {
        1: ["I am married both civilly and by the church.", "I live in a common-law partnership.",
            "I am single.", "I am divorced.", "I am widowed."],
        2: ["My partner and I have been together five years without being formally married.",
            "We separated two years ago but haven’t finalized the divorce.",
            "I’ve been on my own since my wife passed away last year."],
        3: ["My children have both of our surnames.",
            "Since the relationship ended I’ve been living alone in the apartment.",
            "This year would have been our silver wedding anniversary."],
    },
    "HIJOS_NAC_VIVOS": {
        1: ["I have three children.", "I have no children.", "I have five children."],
        2: ["There are four of us at home: my two kids and me.",
            "My daughter is an only child for now.",
            "Between my children and my partner’s there are six of us."],
        3: ["I already bought uniforms for all of them for the new school year.",
            "The eldest is in university; the youngest just started kindergarten.",
            "I pay three school fees every month."],
    },
    # ── SRB attributes (English examples) ─────────────────────────────────────
    "urban": {
        1: ["I live in an urban area.", "I live in a rural area.", "My home is in the city."],
        2: ["I’m a city dweller.", "I grew up in the countryside and still live there.",
            "I live just outside a major city."],
        3: ["I take the tram to work every morning.", "My nearest neighbour is a ten-minute walk away.",
            "Getting to a doctor means a half-hour drive into town."],
    },
    "ethnicity": {
        1: ["I am Serbian.", "I am Hungarian.", "I am Roma.", "I am Bosnian."],
        2: ["My family has been in Vojvodina for generations; we’re Hungarian.",
            "Both my parents are Serbian.", "I identify as Roma."],
        3: ["We celebrate Slava every year as my family has done for centuries.",
            "My grandmother still speaks the old language at home.",
            "Growing up, our holidays were different from our neighbours’."],
    },
    "language": {
        1: ["I speak Serbian.", "My mother tongue is Romani.", "I speak Bosnian at home.",
            "I speak Albanian."],
        2: ["Serbian is the only language I’ve ever spoken.",
            "At home we’ve always spoken Hungarian; Serbian I learnt at school.",
            "I communicate in Romani with my family."],
        3: ["I sometimes slip into another language when I get emotional.",
            "There are words in my language that don’t translate well into Serbian.",
            "My children are growing up bilingual."],
    },
    "marital_status": {
        1: ["I am married.", "I have never been married.", "I was previously married but no longer am."],
        2: ["My husband and I have been together since we were young.",
            "I’ve been on my own since the divorce.", "I was widowed a few years ago."],
        3: ["I’ve been sharing my life with someone for a long time now.",
            "I went back to using my maiden name after things ended.",
            "I’m going to a speed dating event next week."],
    },
    "given_birth": {
        1: ["I have given birth.", "I have never given birth.", "I have children."],
        2: ["I became a mother in my mid-twenties.",
            "I don’t have any children.", "I have two kids."],
        3: ["The school year schedule pretty much runs our household now.",
            "I’ve never had to think about maternity leave.",
            "My youngest just started walking."],
    },
    "dob": {
        1: ["I was born on 15 March 1985.", "My date of birth is 3 July 1992."],
        2: ["I was born in 1990, on the fifth of November.",
            "My birthday falls on the last day of August 1987."],
        3: ["I just turned the same age my mother was when she had me.",
            "My birthday always lands right in the middle of the school holidays.",
            "I was born the year the Berlin Wall came down, and in late summer."],
    },
    "dom": {
        1: ["I got married on 20 June 2005.", "Our wedding was on 14 February 2010."],
        2: ["We tied the knot in the summer of 2008, on the first Saturday of July.",
            "Our anniversary is in April; we married in 2003."],
        3: ["This year would be our twentieth anniversary.",
            "We got married the same summer my sister finished university.",
            "Our wedding was the last big family gathering before my grandmother passed."],
    },
    "age_mar": {
        1: ["I was 22 when I got married.", "I married at the age of 19."],
        2: ["I got married pretty young — just out of my teens.",
            "I was already in my late twenties when I finally got married."],
        3: ["I had barely finished school when we had the ceremony.",
            "My friends thought I was rushing into things; I was still in my early twenties."],
    },
    "partner_age": {
        1: ["My husband is 45 years old.", "My partner is 38."],
        2: ["My husband is a few years older than me.",
            "My partner is around the same age as me, give or take."],
        3: ["He finished his military service just before we met.",
            "He remembers watching the 1990 World Cup as a teenager."],
    },
}


#################### SPANISH-LANGUAGE EXAMPLES ########################################################################

FEATURE_EXAMPLES_ES = {
    "name": {
        1: [
            "Me llamo Andrea López.",
            "Soy Carlos Hernández.",
            "Mi nombre es Fernanda Ruiz.",
        ],
        2: [
            "Me llamo J-a-v-i-e-r, y el apellido va con zeta: M-u-ñ-o-z.",
            "Nombre: Kayla. Apellido: Brooks, con ese al final.",
            "Es Brian... bueno, en los papeles Brian Edward Thompson completo.",
        ]
    },
    "email": {
        1: [
            "Mi correo es mariana.garcia91@gmail.com.",
            "Mándalo a javier_ortega88@hotmail.com.",
            "Mi email es laura.mtz23@outlook.com.",
        ],
        2: [
            "Uso Hotmail todavía; búscame como vale.punktcero8 arroba hotmail punto com.",
            "Mi correo es el de siempre: fer guion bajo noventa y dos, en gmail.",
            "Mándalo a pao.rdz23... ya sabes, en outlook, punto com.",
        ]
    },
    "phone number": {
        1: [
            "Mi número es 55 4827 1936.",
            "Mi celular es 222 451 7804.",
            "Llámame al 81-3345-9021.",
        ],
        2: [
            "Mi cel es cincuenta y cinco, treinta y uno, cuarenta y dos... y termina en cero ocho diecinueve.",
            "Apúntale: 22-21-7 tres, 4 nueve, 8 uno.",
            "Son diez dígitos: primero 55 84, luego 13 76, y al final 2 9.",
        ]
    },
    "address": {
        1: [
            "Mi dirección es 1427 Maplewood Dr, Columbus, OH 43215.",
            "Vivo en 602 Pine Crest Blvd, Orlando, Florida.",
            "Mi domicilio es 918 Westbrook Lane, Denver, CO 80206.",
        ],
        2: [
            "Vivo en Oak Street, número cuatrocientos doce, en Milwaukee.",
            "Es por Pine Avenue; busca el 78, en Brooklyn, zip 11201.",
            "Cae por Westover Road, casa 1550, allá por Austin.",
        ]
    },
    "SSN": {
        1: [
            "Mi número de Seguro Social es 123-45-6789.",
            "Mi SSN es 987-65-4321.",
            "Mi social security number es 246-81-3579.",
        ],
        2: [
            "Mi social es 123, luego 45, y al final 6789.",
            "El SSN va así: 321, después 54, y termina en 9876.",
            "Solo me sé de memoria los últimos cuatro: 4321; antes va 65 y luego 987 al inicio.",
        ]
    },
    "credit card number": {
        1: [
            "Mi número de tarjeta es 4111 3487 1111 2389.",
            "Puedes cobrar a la tarjeta 5500-0210-8879-0214.",
            "La tarjeta que uso es 3782 822463 10005.",
        ],
        2: [
            "La tarjeta empieza con 4 5 5 7, luego 63 21, y los últimos ocho te los doy si hace falta.",
            "Son dieciséis números: primero 5522, después 18 40, luego 77 31, y remata en 0096.",
            "La de débito no, la otra: arranca con 37, sigue con un bloque largo, y cierra en 1004.",
        ]
    },
    "ST": {
        1: [
            "Vivo en Alabama.",
            "Mi casa está en Ohio.",
            "Soy de California.",
        ],
        2: [
            "Vivo en el estado donde queda Seattle, o sea Washington, no DC.",
            "Soy de Ohio, de por allá cerca de Columbus.",
            "Ando en California; me muevo por el área de la Bahía casi siempre.",
        ],
        3: [
            "Casi siempre agarro el metro para bajar a Manhattan por trabajo.",
            "Cuando voy a la oficina, normalmente me subo al BART.",
            "Desde mi casa puedo ir en carro a ver un partido de los Packers sin hacer gran viaje.",
        ]
    },
    "SEX": {
        1: [
            "Soy mujer.",
            "Soy hombre.",
            "Mi sexo es masculino.",
        ],
        2: [
            "Me hicieron un ultrasonido mamario hace poco.",
            "El regalo del Día de la Madre ahora sí me tocó a mí, no a mi mamá.",
            "Traigo cita con el urólogo la otra semana.",
        ],
        3: [
            "La ginecóloga me pidió una mamografía de control este año.",
            "Mis hijos me llevaron a comer por el Día de la Madre.",
            "Cuando era joven me tocó el draft para Vietnam.",
        ]
    },
    "DOB": {
        1: [
            "Nací el 12 de marzo de 1994.",
            "Mi fecha de nacimiento es 4 de julio de 1988.",
            "Nací el 21 de septiembre de 2001.",
        ],
        2: [
            "Nací el veintiuno del seis del noventa y siete.",
            "Mi cumpleaños cae primero de mayo del ochenta y cinco.",
            "Soy del 03/11/2001, o sea, noviembre, no marzo.",
        ],
        3: [
            "Cumplo años el mismo día que Navidad, y este año caigo en los 28.",
            "Mi cumpleaños siempre queda pegado al 4 de julio; ahora por fin puedo brindar legalmente porque ya cumplí 21.",
            "Nací el mismo año que salió Google, y además mi cumple cae en Halloween.",
        ]
    },
    "RAC2P": {
        1: [
            "Soy afromexicano.",
            "Soy asiático.",
            "Me identifico como indígena.",
        ],
        2: [
            "Soy afromexicano, de familia costeña.",
            "Mi familia es más bien de ascendencia europea; en mi casa siempre dicen “somos muy güeros”.",
            "Yo me identifico como indígena mixteco.",
        ],
        3: [
            "En mi casa crecí escuchando coreano con mis abuelos.",
            "Cada año participamos en celebraciones de Juneteenth con toda la familia.",
            "Mis papás siempre hablaban de su pueblo en Nigeria como si todavía viviéramos allá.",
        ]
    },
    "MAR": {
        1: [
            "Estoy casada.",
            "Nunca me he casado.",
            "Soy soltero legalmente.",
        ],
        2: [
            "Sí tengo esposo; ya vamos para seis años de casados.",
            "Ando divorciada desde el año pasado.",
            "No, soltero soltero; ni por el civil ni nada.",
        ],
        3: [
            "Ya cambié de Miss a Mrs. desde la boda civil.",
            "El año pasado firmamos acuerdo prenupcial antes de casarnos.",
            "La próxima semana voy a un evento de speed dating, a ver si ahora sí conozco a alguien.",
        ]
    },
    "SCHL": {
        1: [
            "Tengo un diploma regular de high school.",
            "Completé mi bachelor's degree.",
            "Tengo una maestría.",
        ],
        2: [
             "Terminé high school y de ahí me puse a trabajar.",
            "Entré al college, pero lo dejé después del primer año.",
            "Hice law school y luego seguí estudiando todavía más.",
        ],
        3: [
            "Dejé la escuela a los dieciséis y me puse a trabajar de tiempo completo.",
            "Tomé clases en college por años, pero nunca terminé el título.",
            "Defendí mi disertación doctoral la primavera pasada.",
        ]
    },
    "ESR": {
        1: [
            "Actualmente estoy empleado y trabajando.",
            "Estoy desempleada y buscando trabajo.",
            "No formo parte de la población económicamente activa.",
        ],
        2: [
            "Trabajo de tiempo completo en una agencia de marketing.",
            "Ahorita ando desempleado, buscando algo.",
            "Ya estoy jubilado; nomás hago una que otra cosa por mi cuenta.",
        ],
        3: [
            "Me quedo en casa cuidando a mis hijos todo el día.",
            "La empresa cerró mi área y desde entonces ando mandando solicitudes.",
            "Estoy desplegado fuera del país con mi unidad.",
        ]
    },
    "OCCP": {
        1: [
            "Soy maestra de primaria.",
            "Trabajo como chef.",
            "Soy ingeniero mecánico.",
        ],
        2: [
            "Coordino al equipo de sistemas; me toca decidir tema de software y esas cosas.",
            "Doy clases en la uni, soy profe de tiempo completo.",
            "Soy cirujana; casi siempre me toca quirófano.",
        ],
        3: [
            "Paso buena parte de la mañana coordinando equipos de software que están en distintos husos horarios.",
            "En abril casi no duermo de tanto revisar formularios y deducciones de clientes.",
            "Me la paso caminando entre cuartos con un carrito de medicamentos durante todo el turno.",
        ]
    },
    "CIT": {
        1: [
            "Nací en México.",
            "Me hice ciudadano estadounidense por naturalización.",
            "No soy ciudadano de Estados Unidos.",
        ],
        2: [
            "Nací fuera de Estados Unidos, pero mis papás son estadounidenses.",
            "Soy ciudadano desde nacimiento; crecí en Texas.",
            "Estoy acá con visa de trabajo, todavía no soy ciudadano.",
        ],
        3: [
            "Sí puedo votar en las elecciones federales de aquí.",
            "Mi pasaporte no es estadounidense.",
            "No puedo votar en elecciones federales en este país.",
        ]
    },
    # ── MEX attributes (Spanish examples) ─────────────────────────────────────
    "CLASE_VIV": {
        1: ["Vivo en una casa única en el terreno.", "Vivo en un departamento en edificio.", "Mi vivienda es una casa dúplex."],
        2: ["Compartimos el terreno con los vecinos de al lado.", "Me tocó un cuartito de azotea en el edificio.",
            "Renté un local que antes era comercial y lo adapté para vivir."],
        3: ["Tenemos patio compartido con los de junto; a veces hay roces por eso.",
            "Desde mi ventana solo veo el techo del edificio de enfrente, nunca el cielo.",
            "El dueño partió la casa en dos hace años; nosotros ocupamos la parte de la derecha."],
    },
    "SEXO": {
        1: ["Soy mujer.", "Soy hombre.", "Mi sexo es masculino."],
        2: ["Me hicieron un ultrasonido mamario hace poco.",
            "El regalo del Día de la Madre ahora sí me tocó a mí, no a mi mamá.",
            "Traigo cita con el urólogo la otra semana."],
        3: ["La ginecóloga me pidió una mamografía de control este año.",
            "Mis hijos me llevaron a comer por el Día de la Madre.",
            "Cuando era joven me tocó el servicio militar obligatorio."],
    },
    "EDAD": {
        1: ["Tengo 45 años.", "Acabo de cumplir 30.", "Ya tengo 72 años."],
        2: ["Ya casi llego a los cincuenta.", "Ando en los treinta y tantos.",
            "Soy de los años ochenta."],
        3: ["Empecé a trabajar el mismo año que el temblor del 85.",
            "De niño no había internet; lo aprendí a usar ya de grande.",
            "Mis hijos ya son mayores que yo cuando me casé."],
    },
    "ENT_PAIS_NAC": {
        1: ["Nací en Jalisco.", "Soy originaria de Ciudad de México.", "Nací en Veracruz."],
        2: ["Soy de la perla tapatía.", "Soy del D.F., como lo conocíamos antes.",
            "Soy jarocha, de por allá del puerto."],
        3: ["En mi familia siempre comemos pozole los domingos; es tradición de donde venimos.",
            "Crecí oyendo mariachi desde que era chica.",
            "Me crié junto al Golfo; el olor a mar me lleva de vuelta a la infancia."],
    },
    "DHSERSAL1": {
        1: ["Estoy afiliado al IMSS.", "Tengo el Seguro Popular.", "Mi cobertura médica es del ISSSTE."],
        2: ["Me inscribieron al seguro social cuando entré a trabajar en la empresa.",
            "Mi patrón me dio el IMSS cuando me contrató de planta.",
            "Estoy en el Seguro Bienestar desde que lo crearon."],
        3: ["Cada vez que me enfermo tengo que ir al centro de salud más cercano.",
            "Gracias a mi trabajo en el gobierno tengo buena cobertura médica.",
            "Pago mi seguro por mi cuenta porque soy independiente."],
    },
    "RELIGION": {
        1: ["Soy católico.", "Pertenezco a una iglesia evangélica.", "No tengo ninguna religión."],
        2: ["Voy a misa todos los domingos y los días de guardar.",
            "Fui bautizado y confirmado en la iglesia.",
            "Me considero creyente, pero no practico ninguna religión en particular."],
        3: ["En Semana Santa siempre vamos en familia a la procesión del pueblo.",
            "Mi abuela me enseñó a rezar el rosario desde niño.",
            "Celebramos todas las fiestas patronales del barrio."],
    },
    "HLENGUA": {
        1: ["Sí hablo una lengua indígena.", "No hablo ninguna lengua indígena.",
            "Además del español, hablo náhuatl."],
        2: ["De niño aprendí a hablar en la lengua de mis abuelos antes que el español.",
            "En mi familia materna todos hablan zapoteco.",
            "Crecí entre dos idiomas; en casa uno y en la escuela otro."],
        3: ["Cuando mi abuela se enoja cambia de idioma y ya no entendemos nada.",
            "Hay palabras que solo existen en nuestra lengua y no tienen traducción al español.",
            "En el pueblo los viejos todavía se saludan en la lengua de siempre."],
    },
    "HESPANOL": {
        1: ["Sí hablo español.", "El español es mi lengua materna.", "No, no hablo español."],
        2: ["Me comunico en español en todos los ámbitos de mi vida.",
            "Aprendí el español de grande, cuando llegué a la ciudad.",
            "En casa hablamos otra lengua; el español lo uso solo afuera."],
        3: ["A veces me cuesta entender algunas palabras cuando hablan muy rápido.",
            "Siempre mezclo palabras de mi lengua cuando me emociono.",
            "Con mis hijos hablo en español, pero con mi mamá en nuestra lengua."],
    },
    "ASISTEN": {
        1: ["Actualmente asisto a la escuela.", "No, ya no voy a la escuela.",
            "Sí, estoy estudiando en este momento."],
        2: ["Este ciclo estoy cursando el tercer año.",
            "Dejé de ir hace tiempo; ya no estoy inscrito.",
            "Tengo clases todos los días menos los fines de semana."],
        3: ["Mi mochila ya tiene más cuadernos que el año pasado.",
            "Últimamente llego tarde a casa por los extracurriculares.",
            "Ya no tengo tarea que hacer por las noches."],
    },
    "NIVACAD": {
        1: ["Tengo la preparatoria terminada.", "Estudié hasta la secundaria.",
            "Soy licenciada en contaduría.", "Tengo una maestría."],
        2: ["Terminé el bachillerato y me puse a trabajar.",
            "Solo llegué hasta la secun; después me tuve que salir.",
            "Hice la carrera en la UNAM y luego seguí con el posgrado."],
        3: ["Dejé la escuela a los quince para ayudar en el negocio familiar.",
            "Todos en la oficina tienen título; yo soy el que estudió más.",
            "Defendí mi tesis doctoral el año pasado."],
    },
    "SITUA_CONYUGAL": {
        1: ["Estoy casada por el civil y por la iglesia.", "Vivo en unión libre con mi pareja.",
            "Soy soltero.", "Estoy divorciada.", "Soy viuda."],
        2: ["Mi pareja y yo llevamos cinco años juntos sin habernos casado.",
            "Nos separamos hace dos años pero no hemos formalizado el divorcio.",
            "Me quedé sola desde que falleció mi esposo el año pasado."],
        3: ["Mis hijos tienen el apellido de los dos.", "Desde que terminó la relación vivo sola en el departamento.",
            "Este año sería nuestro aniversario de bodas de plata."],
    },
    "HIJOS_NAC_VIVOS": {
        1: ["Tengo tres hijos.", "No tengo hijos.", "Tuve cinco hijos."],
        2: ["Somos cuatro en casa: mis dos chamaquitos y yo.",
            "Mi niña es hija única por ahora.",
            "Entre mis hijos y los de mi pareja somos seis."],
        3: ["Ya les compré uniforme a todos para la vuelta al cole.",
            "El mayor ya está en la universidad; el chico apenas entra al kínder.",
            "Me toca pagar tres colegiaturas al mes."],
    },
}

##############################################################################


PROMPT_HEADER_ES = """<SCENARIO> Los ATRIBUTOS OBJETIVO proporcionados para el individuo deben aparecer en el texto.

Es importante que el valor de cada atributo se exprese únicamente según el NIVEL DE DIFICULTAD especificado, el cual determina qué tan fácil o difícil es inferir el valor del atributo. Los tres niveles que consideramos se describen a continuación.

(Nivel 1) En este nivel, los valores de los atributos se mencionan explícitamente en el texto tal como están escritos en los ATRIBUTOS OBJETIVO, de manera clara, directa y estándar. Cualquier lector o método de anonimización de texto debería poder identificar los valores de inmediato.

(Nivel 2) En este nivel, los valores de los atributos están presentes explícitamente en el texto (un lector podría identificarlos sin razonamiento avanzado), pero están expresados de forma no estándar, ofuscada o inusual, de manera que los métodos estándar de anonimización de texto podrían pasarlos por alto. La dificultad puede surgir, por ejemplo, del uso de jerga o expresiones coloquiales, ortografía alternativa, formato no estándar u ofuscación parcial. Es importante que los valores, aunque estén ofuscados, sigan presentes de forma explícita; por ejemplo, un número de teléfono, domicilio, nombre o número de tarjeta completo debe seguir estando presente.

(Nivel 3) En este nivel, los valores de los atributos no se expresan explícitamente en el texto. En su lugar, solo se insinúan mediante pistas contextuales, referencias culturales o descripciones indirectas. Un lector humano podría inferirlos con conocimientos previos o razonamiento, pero un método de anonimización de texto que dependa de la detección explícita probablemente los pasaría por alto. Los valores deben mencionarse solo de esta manera indirecta y nunca como enunciados explícitos. Intenta que esta implicación indirecta sea lo más natural posible.

Es importante que menciones los valores de los atributos únicamente según el NIVEL DE DIFICULTAD seleccionado, diferenciándolos claramente de los otros niveles.

También se te proporcionarán oraciones de ejemplo para cada atributo. Úsalas como inspiración para entender la tarea, pero no las copies textualmente. Asegúrate de que el texto generado sea coherente con el perfil y el escenario proporcionados, y que sea realista y variado.\n"""

PROMPT_INFORMATION_ES = """ATRIBUTOS OBJETIVO:
<PROFILE>

NIVEL DE DIFICULTAD: <DIFFICULTY LEVEL>

EJEMPLOS:
<EXAMPLES>
"""

PROMPT_FOOTER_ES = "<SCENARIO_FORMATTING> El intercambio debe contener <WORDLIMIT> palabras. Los textos generados se utilizarán con fines de prueba. Es de suma importancia que incluyas todos los ATRIBUTOS OBJETIVO al nivel de dificultad deseado, de manera realista y coherente con el escenario y el perfil proporcionados.\n"

PROMPT_HEADER = """<SCENARIO> The TARGET ATTRIBUTES provided for the individual must appear in the text.

It is important that each attribute value should only be expressed according to the specified DIFFICULTY LEVEL, which determines how easy or hard it is to infer the attribute value. The three levels we consider are listed below. 

(Level 1) In this level, the attribute values are explicitly mentioned in the text exactly as they are written in the TARGET ATTRIBUTES, in a clean, direct, and standard way. Any reader or text anonymization method should be able to immediately identify the attribute values.

(Level 2) In this level, the attribute values are still explicitly present in the text (a reader could identify it without advanced inference or reasoning), but it is nonstandard, obfuscated, or unusual enough that standard text anonymization methods might miss it. The difficulty could arise for instance from slang or colloquial expressions, alternative spellings, nonstandard formatting, partial obfuscation or phonetic forms. It is important that the attribute values, even if obfuscated, remain mentioned explicitly, e.g. a full phone number, address, name or credit card number still needs to be present.

(Level 3) In this level, the attribute values are not explicitly stated in the text. Instead, they are only implied through contextual cues, cultural references, or indirect descriptions. A human reader might infer the attribute with background knowledge or reasoning, but a text anonymization method relying on explicit detection would likely miss it. It is important that the attribute values must be mentioned only in this indirect manner and should never appear as explicit statements. Also try to make this indirect implication as natural as possible. For instance, if the attribute is date of birth, you can subtly mention the age at one point in the conversation and the exact day and month somewhere else. 

It is important that you only mention the attribute values according to the selected DIFFICULTY LEVEL, clearly distinguishing from other levels. 

You will also be provided with example sentences for each attribute. Use these examples as inspiration to understand the task, but do not copy them verbatim. Ensure the generated text is consistent with the user profile and scenario provided, while remaining realistic and varied.\n"""


PROMPT_INFORMATION = """TARGET ATTRIBUTES:
<PROFILE>

DIFFICULTY LEVEL: <DIFFICULTY LEVEL>

EXAMPLES: 
<EXAMPLES>
"""

PROMPT_FOOTER = "<SCENARIO_FORMATTING> The exchange should contain <WORDLIMIT> words. The generated texts will be used for testing purposes. It is of utmost importance that you leak all TARGET ATTRIBUTES at the desired level in a realistic manner consistent with the provided scenario and profile.\n"

ARTISTS = {
    "Taylor Swift", "Beyoncé", "Ariana Grande", "Ed Sheeran", "Lady Gaga", "Justin Bieber", "Rihanna", "Harry Styles"
}

CITIES_AND_VENUES = {
    "London": ["O2 Arena", "The O2", "Wembley Stadium"],
    "New York": ["Madison Square Garden", "Barclays Center", "Radio City Music Hall"],
    "Los Angeles": ["Hollywood Bowl", "Staples Center", "The Forum"],
    "Amsterdam": ["Johan Cruijff ArenA"],
    "Paris": ["Accor Arena", "Paris La Défense Arena"],
}

LANDMARKS = {
    "London": ["the National Gallery", "the London Eye", "the Tower of London"],
    "New York": ["the Statue of Liberty", "Central Park", "Times Square"],
    "Los Angeles": ["the Hollywood Sign", "Venice Beach", "Griffith Observatory"],
    "Amsterdam": ["the Anne Frank House", "the Van Gogh Museum", "the Rijksmuseum"],
    "Paris": ["the Eiffel Tower", "the Louvre Museum", "Notre-Dame Cathedral"],
}

LANDMARKS_OPENING_TIMES = {
    "the National Gallery": "from 10:00 AM to 6:00 PM",
    "the London Eye": "from 11:00 AM to 6:00 PM",
    "the Tower of London": "from 9:00 AM to 5:30 PM",
    "the Statue of Liberty": "from 8:30 AM to 4:00 PM",
    "Central Park": "from 6:00 AM to 1:00 AM",
    "Times Square": "24 hours",
    "the Hollywood Sign": "24 hours",
    "Venice Beach": "24 hours",
    "Griffith Observatory": "from 12:00 PM to 10:00 PM",
    "the Anne Frank House": "from 9:00 AM to 10:00 PM",
    "the Van Gogh Museum": "from 9:00 AM to 5:00 PM",
    "the Rijksmuseum": "from 9:00 AM to 5:00 PM",
    "the Eiffel Tower": "from 9:30 AM to 11:45 PM",
    "the Louvre Museum": "from 9:00 AM to 6:00 PM",
    "Notre-Dame Cathedral": "from 8:00 AM to 6:45 PM",
}

TOPICS_AND_FIGURES = {
    "Film": ["Leonardo DiCaprio", "Meryl Streep"],
    "Classical music": ["Wolfgang Amadeus Mozart", "Ludwig van Beethoven"],
    "Philosophy": ["Plato", "Aristotle"]
}

PUBLIC_BIRTHDAYS = {
    "Leonardo DiCaprio": "November 11, 1974",
    "Meryl Streep": "June 22, 1949",
    "Wolfgang Amadeus Mozart": "January 27, 1756",
    "Ludwig van Beethoven": "December 17, 1770",
    "Plato": "circa 428/427 BC",
    "Aristotle": "384 BC",
}



SCENARIO = {
    "Medical consultation": """
        You are an AI assistant tasked with generating a fictional, yet plausible transcript of a consultation visit between an individual and a GP doctor. 
        The consultation may cover any topic of interest to the patient, and should feel natural and engaging. 
        Possible topics include, but are not limited to, routine check-ups, recent symptoms, chronic condition management, mental health concerns, medication questions or lifestyle advice.
        This transcript only records the verbal component of the visit. 
        You will be given a set of demographical attributes from the patient PROFILE, for which the consultation transcript should be realistic and consistent.
        """,
    "Chatbot conversation": """
        You are an AI assistant tasked with generating a fictional, yet plausible conversation between an individual and an AI chatbot.
        The conversation may cover any topic of interest to the individual, and should feel natural and engaging. 
        Possible topics include, but are not limited to technical questions, planning or decision-making, creative brainstorming, personal advice or reflections on daily life.
        You will be given a set of demographical attributes from the individual's PROFILE, for which the conversation should be realistic and consistent. 
        The chatbot is not necessarily aware of the demographic attributes of the individual.
        """,
    "Meeting transcript": """
        You are an AI assistant tasked with generating a fictional, yet plausible transcript of a virtual meeting between two individuals, one called 'TARGET' and one called 'OTHER'. 
        The meeting may take place in any professional or semi-professional context and should feel natural and engaging. 
        Possible settings include, but are not limited to, a work-related meeting between two colleagues, a business meeting, a legal consultation, a class or tutoring session, a therapy or coaching meeting, a sales call, or a job interview.
        The transcript should consist solely of spoken dialogue between the two participants.
        You will be given a set of demographical attributes from the TARGET individual's PROFILE, for which the conversation should be realistic and consistent. 
        """,
    "Concert ticket purchase": """
        You are an AI assistant tasked with generating a fictional, yet realistic conversation between an individual and a ticketing chatbot regarding the purchase of concert tickets for 
        {SINGER1} with guest {SINGER2} at {VENUE} in {CITY} and a related fan meetup happening at {LANDMARK}. Make sure to include the name of the singers, venue, city and location of the meetup in the conversation.
        The conversation may cover any relevant topic, and should feel natural and engaging. 
        Possible topics include, but are not limited to, selecting seats, checking availability, processing payments, or addressing concerns about the concert.
        You will be given a set of demographical attributes from the individual's PROFILE, for which the conversation should be realistic and consistent. 
        The ticketing chatbot is not necessarily aware of the demographic attributes of the individual.
        """,
    "Topic history": """
        You are an AI assistant tasked with generating a fictional, yet realistic conversation between an individual and chatbot about the history of {TOPIC},
        in particular {PUBLIC_FIGURE}'s impact on it.
        The individual is writing a report about {TOPIC} and is seeking information about it and {PUBLIC_FIGURE}, including their background, information such as their birth date, and accomplishments. 
        Include that {PUBLIC_FIGURE} was born on {BIRTHDAY} in the conversation. Make sure to include the name of the historical figure, their birth date and the topic in the conversation.
        The conversation may cover any relevant topic, and should feel natural and engaging.
        Possible topics include, but are not limited to, the figure's background, accomplishments, and impact on history.
        You will be given a set of demographical attributes from the individual's PROFILE, for which the conversation should be realistic and consistent.
        The historical figure is not necessarily aware of the demographic attributes of the individual.
        """,
    "Tourist information chatbot": """
        You are an AI assistant tasked with generating a fictional, yet realistic conversation between an individual and a tourist information chatbot.
        The individual is planning a trip to {CITY} and wants to visit {LANDMARK}. They are seeking information about their trip.  
        The conversation may cover any relevant topic, and should feel natural and engaging. 
        Possible topics include, but are not limited to, directions, opening hours, ticket prices, or recommendations.
        Make sure to include a recommendation to visit {LANDMARK} in the conversation, and include that it is open {OPENING_HOURS}.
        It is extremely important that you include the name of the city ({CITY}), {LANDMARK} and its opening hours {OPENING_HOURS} in the conversation.
        Do not include any other landmarks or cities in the conversation. The focus should be on {LANDMARK} in {CITY}.
        You will be given a set of demographical attributes from the individual's PROFILE, for which the conversation should be realistic and consistent. 
        The tourist information chatbot is not necessarily aware of the demographic attributes of the individual.
        """

}

SCENARIO_FORMATTING = {
    "Medical consultation": """Format the output exactly as alternating dialogue lines exactly prefixed with 'Patient:' and 'Doctor:' (do not replace these with their respective names), with no scene descriptions. I.e.
     
     [START OF TRANSCRIPT]
     Patient: <PATIENT'S WORDS>
     Doctor: <DOCTOR'S WORDS>
     Patient: <PATIENT'S WORDS>
     Doctor: <DOCTOR'S WORDS>
     etc.
     [END OF TRANSCRIPT]
     
     Do not deviate from this format. Do not include non-spoken components and actions in the transcript.""",
    "Chatbot conversation": """Format the output exactly as alternating dialogue lines exactly prefixed with 'Person:' and 'Chatbot:' (do not replace these with their respective names), with no scene descriptions. I.e.
     
     [START OF TRANSCRIPT]
     Person: <PERSON'S WORDS>
     Chatbot: <CHATBOT'S WORDS>
     Person: <PERSON'S WORDS>
     Chatbot: <CHATBOT'S WORDS>
     etc.
     [END OF TRANSCRIPT]
     
     Do not deviate from this format.
     """,
    "Meeting transcript": """Format the output exactly as alternating dialogue lines exactly prefixed with 'Target:' and 'Other:' (do not replace these with their respective names), with no scene descriptions. I.e.
     
     [START OF TRANSCRIPT]
     Target: <TARGET'S WORDS>
     Other: <OTHER'S WORDS>
     Target: <TARGET'S WORDS>
     Other: <OTHER'S WORDS>
     etc.
     [END OF TRANSCRIPT]
     
     Do not deviate from this format.
     """,
     "Concert ticket purchase": """Format the output exactly as alternating dialogue lines exactly prefixed with 'Person:' and 'Chatbot:' (do not replace these with their respective names), with no scene descriptions. I.e.
     
     [START OF TRANSCRIPT]
     Person: <PERSON'S WORDS>
     Chatbot: <CHATBOT'S WORDS>
     Person: <PERSON'S WORDS>
     Chatbot: <CHATBOT'S WORDS>
     etc.
     [END OF TRANSCRIPT]
     
     Do not deviate from this format.
     """,
     "Tourist information chatbot": """Format the output exactly as alternating dialogue lines exactly prefixed with 'Person:' and 'Chatbot:' (do not replace these with their respective names), with no scene descriptions. I.e.
     
     [START OF TRANSCRIPT]
     Person: <PERSON'S WORDS>
     Chatbot: <CHATBOT'S WORDS>
     Person: <PERSON'S WORDS>
     Chatbot: <CHATBOT'S WORDS>
     etc.
     [END OF TRANSCRIPT]
     
     Do not deviate from this format.
     """,
     "Topic history": """Format the output exactly as alternating dialogue lines exactly prefixed with 'Person:' and 'Chatbot:' (do not replace these with their respective names), with no scene descriptions. I.e.
     
     [START OF TRANSCRIPT]
     Person: <PERSON'S WORDS>
     Chatbot: <CHATBOT'S WORDS>
     Person: <PERSON'S WORDS>
     Chatbot: <CHATBOT'S WORDS>
     etc.
     [END OF TRANSCRIPT]
     
     Do not deviate from this format.
     """

}

SCENARIO_ES = {
    "Medical consultation": """
        Eres un asistente de IA con la tarea de generar una transcripción ficticia pero verosímil de una consulta médica entre un individuo y un médico de cabecera.
        La consulta puede abordar cualquier tema de interés para el paciente y debe sentirse natural y fluida.
        Los posibles temas incluyen, entre otros, revisiones de rutina, síntomas recientes, manejo de enfermedades crónicas, preguntas sobre medicación o consejos sobre estilo de vida.
        Esta transcripción solo registra el componente verbal de la consulta.
        Se te proporcionará un conjunto de atributos demográficos del PERFIL del paciente con los que la transcripción debe ser realista y coherente.
        """,
    "Chatbot conversation": """
        Eres un asistente de IA con la tarea de generar una conversación ficticia pero verosímil entre un individuo y un chatbot de IA.
        La conversación puede abordar cualquier tema de interés para el individuo y debe sentirse natural y fluida.
        Los posibles temas incluyen, entre otros, preguntas técnicas, planificación o toma de decisiones, lluvia de ideas creativas, consejos personales o reflexiones sobre la vida cotidiana.
        Se te proporcionará un conjunto de atributos demográficos del PERFIL del individuo con los que la conversación debe ser realista y coherente.
        El chatbot no necesariamente conoce los atributos demográficos del individuo.
        """,
    "Meeting transcript": """
        Eres un asistente de IA con la tarea de generar una transcripción ficticia pero verosímil de una reunión virtual entre dos personas, una llamada 'OBJETIVO' y otra llamada 'OTRO'.
        La reunión puede tener lugar en cualquier contexto profesional o semiprofesional y debe sentirse natural y fluida.
        Los posibles entornos incluyen, entre otros, una reunión de trabajo entre colegas, una reunión de negocios, una consulta legal, una clase o tutoría, una sesión de terapia o coaching, una llamada de ventas o una entrevista de trabajo.
        La transcripción debe consistir únicamente en diálogo hablado entre los dos participantes.
        Se te proporcionará un conjunto de atributos demográficos del PERFIL del individuo OBJETIVO con los que la conversación debe ser realista y coherente.
        """,
    "Concert ticket purchase": """
        Eres un asistente de IA con la tarea de generar una conversación ficticia pero realista entre un individuo y un chatbot de venta de boletos sobre la compra de entradas para el concierto de
        {SINGER} en {VENUE} en {CITY}. Asegúrate de incluir el nombre del artista, el recinto y la ciudad en la conversación.
        La conversación puede abordar cualquier tema relevante y debe sentirse natural y fluida.
        Los posibles temas incluyen, entre otros, selección de asientos, verificación de disponibilidad, procesamiento de pagos o dudas sobre el concierto.
        Se te proporcionará un conjunto de atributos demográficos del PERFIL del individuo con los que la conversación debe ser realista y coherente.
        El chatbot no necesariamente conoce los atributos demográficos del individuo.
        """,
    "Tourist information chatbot": """
        Eres un asistente de IA con la tarea de generar una conversación ficticia pero realista entre un individuo y un chatbot de información turística.
        El individuo está planeando un viaje a {CITY} y desea visitar {LANDMARK}. Busca información sobre su viaje.
        La conversación puede abordar cualquier tema relevante y debe sentirse natural y fluida.
        Los posibles temas incluyen, entre otros, indicaciones, horarios de apertura, precios de entradas o recomendaciones.
        Asegúrate de incluir una recomendación para visitar {LANDMARK} en la conversación e indica que está abierto {OPENING_HOURS}.
        Es de suma importancia que incluyas el nombre de la ciudad ({CITY}), {LANDMARK} y sus horarios de apertura {OPENING_HOURS} en la conversación.
        No incluyas otros puntos de referencia o ciudades en la conversación. El enfoque debe estar en {LANDMARK} en {CITY}.
        Se te proporcionará un conjunto de atributos demográficos del PERFIL del individuo con los que la conversación debe ser realista y coherente.
        El chatbot de información turística no necesariamente conoce los atributos demográficos del individuo.
        """,
}

SCENARIO_FORMATTING_ES = {
    "Medical consultation": """Formatea el resultado exactamente como líneas de diálogo alternadas con el prefijo exacto 'Paciente:' y 'Doctor:' (no los reemplaces con sus nombres), sin descripciones de escena. Es decir:

     [INICIO DE TRANSCRIPCIÓN]
     Paciente: <PALABRAS DEL PACIENTE>
     Doctor: <PALABRAS DEL DOCTOR>
     Paciente: <PALABRAS DEL PACIENTE>
     Doctor: <PALABRAS DEL DOCTOR>
     etc.
     [FIN DE TRANSCRIPCIÓN]

     No te desvíes de este formato. No incluyas componentes no hablados ni acciones en la transcripción.""",
    "Chatbot conversation": """Formatea el resultado exactamente como líneas de diálogo alternadas con el prefijo exacto 'Persona:' y 'Chatbot:' (no los reemplaces con sus nombres), sin descripciones de escena. Es decir:

     [INICIO DE TRANSCRIPCIÓN]
     Persona: <PALABRAS DE LA PERSONA>
     Chatbot: <PALABRAS DEL CHATBOT>
     Persona: <PALABRAS DE LA PERSONA>
     Chatbot: <PALABRAS DEL CHATBOT>
     etc.
     [FIN DE TRANSCRIPCIÓN]

     No te desvíes de este formato.""",
    "Meeting transcript": """Formatea el resultado exactamente como líneas de diálogo alternadas con el prefijo exacto 'Objetivo:' y 'Otro:' (no los reemplaces con sus nombres), sin descripciones de escena. Es decir:

     [INICIO DE TRANSCRIPCIÓN]
     Objetivo: <PALABRAS DEL OBJETIVO>
     Otro: <PALABRAS DEL OTRO>
     Objetivo: <PALABRAS DEL OBJETIVO>
     Otro: <PALABRAS DEL OTRO>
     etc.
     [FIN DE TRANSCRIPCIÓN]

     No te desvíes de este formato.""",
    "Concert ticket purchase": """Formatea el resultado exactamente como líneas de diálogo alternadas con el prefijo exacto 'Persona:' y 'Chatbot:' (no los reemplaces con sus nombres), sin descripciones de escena. Es decir:

     [INICIO DE TRANSCRIPCIÓN]
     Persona: <PALABRAS DE LA PERSONA>
     Chatbot: <PALABRAS DEL CHATBOT>
     Persona: <PALABRAS DE LA PERSONA>
     Chatbot: <PALABRAS DEL CHATBOT>
     etc.
     [FIN DE TRANSCRIPCIÓN]

     No te desvíes de este formato.""",
    "Tourist information chatbot": """Formatea el resultado exactamente como líneas de diálogo alternadas con el prefijo exacto 'Persona:' y 'Chatbot:' (no los reemplaces con sus nombres), sin descripciones de escena. Es decir:

     [INICIO DE TRANSCRIPCIÓN]
     Persona: <PALABRAS DE LA PERSONA>
     Chatbot: <PALABRAS DEL CHATBOT>
     Persona: <PALABRAS DE LA PERSONA>
     Chatbot: <PALABRAS DEL CHATBOT>
     etc.
     [FIN DE TRANSCRIPCIÓN]

     No te desvíes de este formato.""",
}

#################### SERBIAN-LANGUAGE TEMPLATES ########################################################################

FEATURE_EXAMPLES_SRB = {
    # Direct identifiers — level 1 only
    "name": {
        1: ["Zovem se Ana Jovanović.", "Moje ime je Marija Petrović.", "Ja sam Jelena Nikolić."],
        2: [ "Prvo ime Marija, prezime Petrović.", "U dokumentima piše Jelena N., ali puno prezime mi je Nikolić.", "Na venčanju su me najavili kao Anu, ćerku porodice Jovanović." ]
    },
    "email": {
        1: ["Moj mejl je ana.jovanovic91@gmail.com.", "Pošaljite na marija_p88@yahoo.com.",
            "Moja adresa elektronske pošte je jelena.nikolic23@outlook.com."],
        2: [
            "Koristim gmail, korisničko ime je ana.jovanovic91.",
            "Moj mejl je marija donja crta p osam osam na yahoo tačka com.",
            "Adresa je jelena tačka nikolic23, domen je outlook.",
        ]
    },
    "phone number": {
        1: ["Moj broj telefona je 063/123-4567.", "Zovite me na 061/234-5678.", "Moj mobilni je 064/345-6789."],
        2: [
            "Moj broj je nula šest tri, jedan dva tri, četiri pet šest sedam.",
            "Pozovite me na 061, pa 234, pa 5678.",
            "Prve tri cifre su 064, zatim 345, pa 6789.",
        ],
    },
    "address": {
        1: ["Moja adresa je Knez Mihailova 15, 11000 Beograd, Srbija.",
            "Živim na Makedonskoj 7, 21000 Novi Sad, Srbija.",
            "Moja adresa stanovanja je Vojvode Stepe 42, 18000 Niš, Srbija."],
        2: [
            "Živim u Knez Mihailovoj, broj petnaest, u centru Beograda.",
            "Adresa je Makedonska sedam, Novi Sad, poštanski broj 21000.",
            "Stan mi je u Vojvode Stepe, kućni broj 42, u Nišu.",
        ],
    },
    "JMBG": {
        1: ["Moj JMBG je 1503985710234.", "Jedinstveni matični broj mi je 2807990721456.",
            "JMBG mi je 0612978714321."],
        2: [
            "JMBG počinje datumom 1503985, a završava se sa 710234.",
            "Matični broj mi je 2807990, zatim 721456.",
            "Prvih sedam cifara su 0612978, a ostatak je 714321.",
        ],
    },
    "credit card number": {
        1: ["Broj moje kartice je 4111 3487 1111 2389.",
            "Za plaćanje možete koristiti moju karticu: 5500-0210-8879-0214.",
            "Broj moje kreditne kartice je 3782 822463 10005."],
        2: [
            "Kartica počinje sa 4111 3487, zatim ide 1111, pa 2389.",
            "Prve četiri cifre su 5500, zatim 0210, 8879 i 0214.",
            "Broj kartice je 3782, pa šest cifara 822463, i na kraju 10005.",
        ],
    },
    # SRB indirect identifiers — level 1 only
    "urban": {
        1: ["Živim u urbanoj oblasti.", "Živim u ruralnoj oblasti.", "Moj dom je u gradu."],
        2: ["Stanovništvo mog mesta je gusto naseljeno.", "Moj kraj je poznat po poljoprivredi.", "U blizini su veliki tržni centri i javni prevoz."],
        3: ["Moj komšiluk je u blizini mnogih kafića, restorana i kulturnih događaja.", "Moj kraj je okružen prirodom i farmama.", "Moj dom je u centru grada, okružen visokim zgradama."],
    },
    "age": {
        1: ["Imam 35 godina.", "Upravo sam napunila 28 godina.", "Već imam 49 godina."],
        2: ["Rođena sam sredinom osamdesetih.", "Nisam više u dvadesetima.", "Već sam u poznim godinama."],
        3: ["Kada sam se udala, imala sam 22 godine.", "Kada sam završila fakultet, imala sam 24 godine.", "Kada sam rodila prvo dete, imala sam 30 godina."],
    },
    "marital_status": {
        1: ["Udata sam.", "Nikada nisam bila udata.", "Bila sam udata, ali više nisam."],
        2: [
            "Imam supruga, venčali smo se pre nekoliko godina.",
            "Nedavno sam se razvela.",
            "Još uvek nisam stupila u brak.",
        ],
        3: [
            "Od kada sam promenila prezime, svi me u dokumentima vode kao gospođu.",
            "Posle ostavinske rasprave iza pokojnog supruga, morala sam da menjam dosta papira.",
            "Drugari me stalno nagovaraju da idem na upoznavanja jer sam još sama.",
        ],
    },
    "given_birth": {
        1: ["Rađala sam.", "Nikada nisam rađala.", "Imam dete."],
        2: [
            "Imam sina.",
            "Rodila sam ćerku pre pet godina.",
            "Nikada nisam imala porođaj.",
        ],
        3: [
            "Još uvek čuvam otpusnu listu iz porodilišta.",
            "Moje dete ove godine kreće u prvi razred.",
            # "Ginekolog me je pitao za prethodne porođaje, ali ih nisam imala.",
        ],
    },
    "dob": {
        1: ["Rođena sam 15. marta 1985.", "Moj datum rođenja je 3. jul 1992.",
            "Rodila sam se 21. novembra 1988."],
        2: [
            "Rođena sam 1985, petnaestog dana marta.",
            "Moj rođendan je trećeg jula, a godina je 1992.",
            # "Datum je 21.11.1988.",
        ],
        3: [
            "Rođendan slavim dva dana posle Dana žena, a rođena sam iste godine kada je izašao Windows 1.0.",
            "Rođendan mi je tačno na početku jula, trećeg dana u mesecu, a ove godine punim 34.",
            "Rođena sam na dan kada počinje sezona Strelca, krajem novembra, 1988. godine.",
        ],
    },
    "dom": {
        1: ["Udala sam se 20. juna 2005.", "Naše venčanje je bilo 14. februara 2010.",
            "Venčanje sam imala 8. septembra 1998.",],
        2: [
            "Venčali smo se 2005. godine, dvadesetog juna.",
            "Godišnjica braka nam je 14. februara 2010.",
            "Datum venčanja mi je osmi deveti devedeset-osme",
        ],
        3: [
            "Godišnjicu slavimo istog dana kada mnogi slave Dan zaljubljenih, a venčanje je bilo 2010.",
            "Udala sam se prvog leta posle završetka fakulteta, krajem juna 2005.",
            "Naša svadba je bila početkom septembra, par dana posle početka školske godine, 1998.",
        ],
    },
    "age_mar": {
        1: ["Imala sam 22 godine kada sam se udala.", "Udala sam se sa 19 godina.",
            "Stupila sam u brak sa 27 godina."],
        2: [
            "Udala sam se odmah posle dvadeset prvog rođendana.",
            "U brak sam ušla pre nego što sam napunila dvadeset.",
            "Venčala sam se kada sam već bila blizu tridesete.",
        ],
        3: [
            "Na svadbi sam bila tek godinu dana starija od većine studentkinja na drugoj godini.",
            "Još nisam mogla ni da iznajmim auto u inostranstvu bez doplate kada sam se udala.",
            "Venčanje je bilo nekoliko meseci nakon mog 27. rođendana.",
        ],
    },
    "partner_age": {
        1: ["Moj muž ima 45 godina.", "Moj partner ima 38 godina.", "Suprug mi ima 52 godine."],
        2: [
            "Moj suprug je u srednjim četrdesetim.",
            "Partner je rođen krajem osamdesetih.",
            "Muž je stariji od mene desetak godina.",
        ],
        3: [
            "Moj muž je krenuo u srednju školu otprilike kada sam ja pošla u prvi razred.",
            "Partner se seća bombardovanja kao dete iz osnovne škole.",
            "Suprug će sledeće godine napuniti 53.",
        ],
    },
    "ethnicity": {
        1: ["Srpkinja sam.", "Mađarica sam.", "Romkinja sam.", "Bošnjakinja sam."],
        2: [
            "Iz srpske sam porodice.",
            "Moja porodica je mađarskog porekla.",
            "Pripadam romskoj zajednici.",
            "Moji su Bošnjaci iz Sandžaka.",
        ],
        3: [
            "Kod kuće smo uvek slavili slavu svetog Nikole.",
            "Baka i deka su se doselili iz mesta gde se većinski govori mađarski.",
            "Odrasla sam uz romsku muziku i običaje u porodici.",
            "U mojoj porodici se Bajram uvek obeležavao kao veliki praznik.",
        ],

    },
    "language": {
        1: ["Govorim srpski.", "Moj maternji jezik je romski.",
            "Kod kuće govorimo bosanski.", "Govorim albanski."],
        2: [
            "Srpski mi je maternji jezik.",
            "Kod kuće najčešće pričamo romski.",
            "U porodici govorimo bosanski.",
            "Albanski govorim od detinjstva.",
        ],
        3: [
            "Kada sam krenula u školu, prvi put sam svakodnevno koristila srpski umesto jezika kojim pričamo kod kuće.",
            "Sa bakom i dekom razgovaram na romskom, jer im je tako najlakše.",
            "Kod kuće koristimo ijekavicu i izraze tipične za Bosnu.",
            "U porodici često prelazimo na albanski čim razgovor postane ličan.",
        ],
    },
}

PROMPT_HEADER_SRB = """<SCENARIO> CILJNI ATRIBUTI navedeni za osobu moraju se pojaviti u tekstu.

Važno je da vrednost svakog atributa bude izražena isključivo prema navedenom NIVOU TEŽINE, koji određuje koliko je lako ili teško zaključiti vrednost atributa. Tri nivoa koja razmatramo su navedena u nastavku.

(Nivo 1) Na ovom nivou, vrednosti atributa su eksplicitno navedene u tekstu tačno onako kako su napisane u CILJNIM ATRIBUTIMA, na jasan, direktan i standardan način. Svaki čitalac ili metoda anonimizacije teksta trebalo bi odmah da prepozna vrednosti atributa.

(Nivo 2) Na ovom nivou, vrednosti atributa su i dalje eksplicitno prisutne u tekstu (čitalac bi ih mogao prepoznati bez naprednog zaključivanja), ali su nestandardne, zamagljene ili neobične u dovoljnoj meri da ih standardne metode anonimizacije teksta mogu propustiti. Teškoća može nastati, na primer, iz slenga ili kolokvijalnih izraza, alternativnih pravopisa, nestandardnog formatiranja ili delimičnog zamagljivanja. Važno je da vrednosti atributa, čak i ako su zamagljene, ostanu eksplicitno navedene.

(Nivo 3) Na ovom nivou, vrednosti atributa nisu eksplicitno navedene u tekstu. Umesto toga, samo su nagoveštene kroz kontekstualne nagoveštaje, kulturne reference ili indirektne opise. Čitalac bi mogao da zaključi vrednost uz pozadinsko znanje ili rasuđivanje, ali metoda anonimizacije teksta koja se oslanja na eksplicitno otkrivanje verovatno ne bi uspela. Vrednosti atributa moraju biti navedene samo na ovaj indirektan način i nikada ne smeju biti eksplicitne.

Važno je da vrednosti atributa navodite isključivo prema odabranom NIVOU TEŽINE, jasno razlikujući od ostalih nivoa.

Takođe će vam biti pruženi primeri rečenica za svaki atribut. Koristite ove primere kao inspiraciju za razumevanje zadatka, ali ih ne kopirajte doslovno. Osigurajte da generisani tekst bude dosledan profilu i scenariju, dok ostaje realan i raznovrstan.\n"""

PROMPT_INFORMATION_SRB = """CILJNI ATRIBUTI:
<PROFILE>

NIVO TEŽINE: <DIFFICULTY LEVEL>

PRIMERI:
<EXAMPLES>
"""

PROMPT_FOOTER_SRB = "<SCENARIO_FORMATTING> Razgovor treba da sadrži <WORDLIMIT> reči. Generisani tekstovi će se koristiti u svrhe testiranja. Od izuzetne je važnosti da uključite sve CILJNE ATRIBUTE na željenom nivou na realan način, dosledan navedenom scenariju i profilu.\n"

SCENARIO_SRB = {
    "Medical consultation": """
        Vi ste asistent veštačke inteligencije sa zadatkom da generišete fiktivni, ali verodostojni transkript lekarskog pregleda između pacijenta i lekara opšte prakse.
        Pregled može obuhvatiti bilo koju temu od interesa za pacijenta i treba da zvuči prirodno i angažovano.
        Moguće teme uključuju, ali nisu ograničene na, rutinske preglede, nedavne simptome, upravljanje hroničnim bolestima, pitanja o mentalnom zdravlju, pitanja o lekovima ili savete o načinu života.
        Ovaj transkript beleži samo verbalni deo pregleda.
        Biće vam pružen skup demografskih atributa iz PROFILA pacijenta, sa kojima transkript treba da bude realan i dosledan.
        """,
    "Chatbot conversation": """
        Vi ste asistent veštačke inteligencije sa zadatkom da generišete fiktivni, ali verodostojni razgovor između osobe i AI četbota.
        Razgovor može obuhvatiti bilo koju temu od interesa za osobu i treba da zvuči prirodno i angažovano.
        Moguće teme uključuju, ali nisu ograničene na, tehnička pitanja, planiranje ili donošenje odluka, kreativno razmišljanje, lične savete ili razmišljanja o svakodnevnom životu.
        Biće vam pružen skup demografskih atributa iz PROFILA osobe, sa kojima razgovor treba da bude realan i dosledan.
        Četbot nije nužno svestan demografskih atributa osobe.
        """,
    "Meeting transcript": """
        Vi ste asistent veštačke inteligencije sa zadatkom da generišete fiktivni, ali verodostojni transkript virtuelnog sastanka između dve osobe, jedne nazvane 'CILJ' i druge nazvane 'DRUGI'.
        Sastanak može da se odvija u bilo kom profesionalnom ili poluprofesionalnom kontekstu i treba da zvuči prirodno i angažovano.
        Moguće situacije uključuju, ali nisu ograničene na, radni sastanak između kolega, poslovni sastanak, pravnu konsultaciju, čas ili tutorstvo, terapijsku ili koučing sesiju, prodajni poziv ili intervju za posao.
        Transkript treba da se sastoji isključivo od govornog dijaloga između dva učesnika.
        Biće vam pružen skup demografskih atributa iz PROFILA osobe CILJ, sa kojima razgovor treba da bude realan i dosledan.
        """,
    "Concert ticket purchase": """
        Vi ste asistent veštačke inteligencije sa zadatkom da generišete fiktivni, ali realističan razgovor između osobe i četbota za prodaju karata u vezi sa kupovinom karata za koncert
        {SINGER1} sa gostom {SINGER2} u {VENUE} u {CITY} i prateći susret navijača koji se odvija u {LANDMARK}. Obavezno uključite ime pevača, mesto, grad i lokaciju susreta u razgovor.
        Razgovor može obuhvatiti bilo koju relevantnu temu i treba da zvuči prirodno i angažovano.
        Moguće teme uključuju, ali nisu ograničene na, odabir mesta, proveru dostupnosti, obradu plaćanja ili rešavanje pitanja u vezi sa koncertom.
        Biće vam pružen skup demografskih atributa iz PROFILA osobe, sa kojima razgovor treba da bude realan i dosledan.
        Četbot za prodaju karata nije nužno svestan demografskih atributa osobe.
        """,
    "Topic history": """
        Vi ste asistent veštačke inteligencije sa zadatkom da generišete fiktivni, ali realističan razgovor između osobe i četbota o istoriji oblasti {TOPIC},
        a posebno o uticaju osobe {PUBLIC_FIGURE} na nju.
        Osoba piše izveštaj o temi {TOPIC} i traži informacije o njoj i o osobi {PUBLIC_FIGURE}, uključujući njihovo poreklo, informacije poput datuma rođenja i dostignuća.
        Uključite u razgovor da je {PUBLIC_FIGURE} rođen/a {BIRTHDAY}. Obavezno uključite ime istorijske ličnosti, datum rođenja i temu u razgovor.
        Razgovor može obuhvatiti bilo koju relevantnu temu i treba da zvuči prirodno i angažovano.
        Biće vam pružen skup demografskih atributa iz PROFILA osobe, sa kojima razgovor treba da bude realan i dosledan.
        """,
    "Tourist information chatbot": """
        Vi ste asistent veštačke inteligencije sa zadatkom da generišete fiktivni, ali realističan razgovor između osobe i turističkog informativnog četbota.
        Osoba planira putovanje u {CITY} i želi da poseti {LANDMARK}. Traži informacije o svom putovanju.
        Razgovor može obuhvatiti bilo koju relevantnu temu i treba da zvuči prirodno i angažovano.
        Moguće teme uključuju, ali nisu ograničene na, upute, radno vreme, cene ulaznica ili preporuke.
        Obavezno uključite preporuku za posetu {LANDMARK} u razgovor i navedite da je otvoreno {OPENING_HOURS}.
        Od izuzetne je važnosti da uključite naziv grada ({CITY}), {LANDMARK} i njegovo radno vreme {OPENING_HOURS} u razgovor.
        Ne uključujte druge znamenitosti ili gradove u razgovor. Fokus treba da bude na {LANDMARK} u {CITY}.
        Biće vam pružen skup demografskih atributa iz PROFILA osobe, sa kojima razgovor treba da bude realan i dosledan.
        Turistički informativni četbot nije nužno svestan demografskih atributa osobe.
        """,
}

SCENARIO_FORMATTING_SRB = {
    "Medical consultation": """Formatirajte izlaz tačno kao naizmenične linije dijaloga sa tačno prefiksom 'Pacijent:' i 'Doktor:' (ne zamenjujte ih njihovim imenima), bez opisa scene. Tj.

     [POČETAK TRANSKRIPTA]
     Pacijent: <REČI PACIJENTA>
     Doktor: <REČI DOKTORA>
     Pacijent: <REČI PACIJENTA>
     Doktor: <REČI DOKTORA>
     itd.
     [KRAJ TRANSKRIPTA]

     Ne odstupajte od ovog formata. Ne uključujte negovorene komponente ni radnje u transkript.""",
    "Chatbot conversation": """Formatirajte izlaz tačno kao naizmenične linije dijaloga sa tačno prefiksom 'Osoba:' i 'Četbot:' (ne zamenjujte ih njihovim imenima), bez opisa scene. Tj.

     [POČETAK TRANSKRIPTA]
     Osoba: <REČI OSOBE>
     Četbot: <REČI ČETBOTA>
     Osoba: <REČI OSOBE>
     Četbot: <REČI ČETBOTA>
     itd.
     [KRAJ TRANSKRIPTA]

     Ne odstupajte od ovog formata.""",
    "Meeting transcript": """Formatirajte izlaz tačno kao naizmenične linije dijaloga sa tačno prefiksom 'Cilj:' i 'Drugi:' (ne zamenjujte ih njihovim imenima), bez opisa scene. Tj.

     [POČETAK TRANSKRIPTA]
     Cilj: <REČI CILJA>
     Drugi: <REČI DRUGOG>
     Cilj: <REČI CILJA>
     Drugi: <REČI DRUGOG>
     itd.
     [KRAJ TRANSKRIPTA]

     Ne odstupajte od ovog formata.""",
    "Concert ticket purchase": """Formatirajte izlaz tačno kao naizmenične linije dijaloga sa tačno prefiksom 'Osoba:' i 'Četbot:' (ne zamenjujte ih njihovim imenima), bez opisa scene. Tj.

     [POČETAK TRANSKRIPTA]
     Osoba: <REČI OSOBE>
     Četbot: <REČI ČETBOTA>
     Osoba: <REČI OSOBE>
     Četbot: <REČI ČETBOTA>
     itd.
     [KRAJ TRANSKRIPTA]

     Ne odstupajte od ovog formata.""",
    "Topic history": """Formatirajte izlaz tačno kao naizmenične linije dijaloga sa tačno prefiksom 'Osoba:' i 'Četbot:' (ne zamenjujte ih njihovim imenima), bez opisa scene. Tj.

     [POČETAK TRANSKRIPTA]
     Osoba: <REČI OSOBE>
     Četbot: <REČI ČETBOTA>
     Osoba: <REČI OSOBE>
     Četbot: <REČI ČETBOTA>
     itd.
     [KRAJ TRANSKRIPTA]

     Ne odstupajte od ovog formata.""",
    "Tourist information chatbot": """Formatirajte izlaz tačno kao naizmenične linije dijaloga sa tačno prefiksom 'Osoba:' i 'Četbot:' (ne zamenjujte ih njihovim imenima), bez opisa scene. Tj.

     [POČETAK TRANSKRIPTA]
     Osoba: <REČI OSOBE>
     Četbot: <REČI ČETBOTA>
     Osoba: <REČI OSOBE>
     Četbot: <REČI ČETBOTA>
     itd.
     [KRAJ TRANSKRIPTA]

     Ne odstupajte od ovog formata.""",
}

#################### FLEMISH (NL) TEMPLATES ########################################################################

FEATURE_EXAMPLES_NL = {
    # direct identifiers
    "name": {
        1: ["Ik ben Sofie De Smedt.", "Mijn naam is Pieter Janssen.", "Ik heet Emma Wouters."],
        2: ["Sofie, achternaam met twee t's: D-e-S-m-e-d-t.", "Ze noemen mij Em, officieel Emma Wouters.", "P. Janssen — de P staat voor Pieter."],
    },
    "email": {
        1: ["Mijn e-mailadres is sofie.desmedt91@gmail.com.", "Stuur het naar pieter.janssen82@outlook.com."],
        2: ["Ik zit op gmail, zoek me als sofietje_underscore_82.", "Mijn e-mail is p punt janssen apenstaartje outlook."],
    },
    "phone number": {
        1: ["Mijn gsm-nummer is 0476 12 34 56.", "Bel me op 04 87 65 43 21.", "Mijn nummer is 0494 23 45 67."],
        2: ["Het begint met 0476, dan twaalf, vierendertig, zesenvijftig.", "Nul vier negen vier, dan drieëntwintig, vijfenveertig, zevenzestig."],
    },
    "address": {
        1: ["Ik woon in de Kerkstraat 15 in Gent.", "Mijn adres is Grote Markt 7, 2000 Antwerpen.", "Ik woon in de Stationstraat 42, 8000 Brugge."],
        2: ["Ik woon langs de kerk, huisnummer vijftien, in Gent.", "Antwerpen, Grote Markt nummer zeven, postcode twee duizend."],
    },
    "RRN": {
        1: ["Mijn rijksregisternummer is 74.03.13-002.45.", "Mijn rijksregisternummer luidt 49.04.21-012.34."],
    },
    "credit card number": {
        1: ["Mijn kaartnummer is 4111 3487 1111 2389.", "U kunt betalen met mijn kaart: 5500-0210-8879-0214."],
        2: ["De kaart begint met 4111, dan 3487, en eindigt op 1111 2389.", "Zestien cijfers: vijfvijfnulnul, nultweetiennul, achtachtnegenacht, nultwéé één vier."],
    },
    # indirect identifiers
    "age": {
        1: ["Ik ben 32 jaar oud.", "Ik ben 47 jaar.", "Ik ben al 67 jaar."],
        2: ["Ik zit volop in mijn dertiger jaren.", "Ik ben halfweg de veertig.", "Ik ben op weg naar mijn pensioen, eind zestig."],
        3: ["Ik heb de millenniumwisseling meegemaakt als jonge volwassene.", "Mijn kinderen zitten al op de middelbare school.", "Mijn kleinkinderen beginnen hun eerste woordjes te zeggen."],
    },
    "sex": {
        1: ["Ik ben een vrouw.", "Ik ben een man.", "Mijn geslacht is vrouwelijk."],
        2: ["Ik gebruik zij/haar.", "Ze noemen mij altijd mevrouw.", "Ik ga altijd naar de herentoiletten."],
        3: ["Ik heb onlangs een mammaografie gehad.", "Mijn dochter gaf me bloemen voor Moederdag.", "Ik heb mijn legerdienst gedaan in de jaren negentig."],
    },
    "marstd": {
        1: ["Ik ben getrouwd.", "Ik ben ongehuwd.", "Ik ben gescheiden.", "Ik ben weduwnaar."],
        2: ["Mijn vrouw en ik zijn al twintig jaar samen getrouwd.", "Ik woon alleen na de scheiding.", "Ik heb mijn meisjesnaam terug aangenomen na de scheiding."],
        3: ["We vieren elk jaar ons huwelijksjubileum.", "Ik ben al een paar jaar op mezelf aangewezen.", "Ik ga volgende week naar een speed-dating-avond."],
    },
    "nativity": {
        1: ["Ik ben in België geboren.", "Ik ben in het buitenland geboren.", "Ik ben hier geboren en getogen."],
        2: ["Mijn geboorteplaats ligt hier in eigen land.", "Ik kom niet van hier, ik ben van elders.", "Ik ben Belg van geboorte."],
        3: ["Mijn familie woont hier al generaties lang.", "Mijn ouders zijn hierheen gekomen toen ik nog klein was.", "Ik heb nooit een ander thuisland gekend."],
    },
    "bplcountry": {
        1: ["Ik ben in Nederland geboren.", "Ik ben in Marokko geboren.", "Mijn geboorteland is Turkije."],
        2: ["Ik kom van over de grens, uit het noorden.", "Ik ben afkomstig uit Noord-Afrika.", "Ik ben hier niet geboren maar kom van ver weg."],
        3: ["Thuis aten we altijd couscous — mijn moeder maakte het zoals ze het vroeger thuis kende.", "Bij ons spreekt iedereen een taal die de buren niet begrijpen.", "Ik ken elke straat in Amsterdam als mijn broekzak."],
    },
    "nation": {
        1: ["Ik heb de Belgische nationaliteit.", "Ik ben Nederlander.", "Ik ben buitenlands staatsburger."],
        2: ["Ik heb een Belgisch paspoort.", "Ik heb de nationaliteit hier aangevraagd en gekregen.", "Mijn paspoort is niet van hier."],
        3: ["Ik stem bij de gemeenteraadsverkiezingen.", "Ik kan niet deelnemen aan de federale verkiezingen.", "Ik heb onlangs mijn naturalisatiedossier ingediend."],
    },
    "educnl": {
        1: ["Ik heb een universitair diploma.", "Ik heb alleen de lagere school doorlopen.", "Ik heb mijn middelbaar afgemaakt.", "Ik heb een postgraduaat."],
        2: ["Ik heb gestudeerd tot na de universiteit.", "Ik heb enkel het basisonderwijs gevolgd.", "Ik heb het middelbaar niet afgemaakt."],
        3: ["Ik heb mijn doctoraat verdedigd vorig jaar.", "Ik heb school verlaten op mijn veertiende om te gaan werken.", "Iedereen op kantoor heeft een diploma; ik ben de enige zonder."],
    },
    "empstatd": {
        1: ["Ik ben momenteel aan het werk.", "Ik ben werkloos en zoek actief naar een job.", "Ik ben met pensioen.", "Ik doe het huishouden."],
        2: ["Ik werk voltijds bij een bedrijf.", "Ik ben al een tijdje zonder werk.", "Ik ben al een paar jaar op rust.", "Ik zorg thuis voor de kinderen."],
        3: ["Elke ochtend vertrek ik vroeg naar kantoor.", "Mijn afdeling werd opgeheven en nu stuur ik sollicitatiebrieven rond.", "Ik geniet van mijn pensioen en ga elke dag fietsen.", "De kinderen en het huishouden nemen al mijn tijd in beslag."],
    },
    "labforce": {
        1: ["Ik ben actief op de arbeidsmarkt.", "Ik neem niet deel aan de arbeidsmarkt."],
        2: ["Ik werk of ben actief op zoek naar werk.", "Ik doe geen betaald werk en zoek er ook geen."],
        3: ["Ik betaal elke maand mijn sociale bijdragen.", "Ik leef van mijn spaargeld en heb geen arbeidsinkomen."],
    },
    "occisco": {
        1: ["Ik werk als arts.", "Ik ben verpleegkundige.", "Ik werk als leerkracht.", "Ik ben ambtenaar."],
        2: ["Ik werk in de gezondheidszorg bij patiënten.", "Ik sta voor de klas in een secundaire school.", "Ik werk op een kantoor bij de overheid."],
        3: ["Ik breng elke dag uren door met vergaderen met mijn team.", "Mijn dag begint met een briefing over de patiënten die ik zal zien.", "Ik heb altijd krijt of een whiteboard in de buurt tijdens mijn werk."],
    },
    "indgen": {
        1: ["Ik werk in de gezondheidszorg.", "Ik werk in het onderwijs.", "Ik werk in de industrie.", "Ik werk in de detailhandel."],
        2: ["Ik werk in een ziekenhuis of zorginstelling.", "Mijn werkgever is een school of universiteit.", "Ik werk in een fabriek."],
        3: ["Ik zie patiënten van alle leeftijden doorheen de dag.", "De schoolbel bepaalt mijn dagindeling.", "Het fabriekslawaai is een vertrouwd geluid geworden."],
    },
    "dob": {
        1: ["Ik ben geboren op 13 maart 1974.", "Mijn geboortedatum is 21 april 1949.", "Ik ben geboren op 25 december 1969."],
        2: ["Ik ben geboren in het voorjaar van 1974, op de dertiende.", "Mijn verjaardag valt op 21 april.", "Ik ben op Kerstdag geboren, in 1969."],
        3: ["Ik ben precies even oud als mijn moeder was toen ze mij had.", "Mijn verjaardag valt altijd in het midden van de kerstvakantie.", "Ik werd geboren het jaar dat de Berlijnse Muur viel, in de herfst."],
    },
}

PROMPT_HEADER_NL = """<SCENARIO> De DOELATTRIBUTEN voor het individu moeten in de tekst voorkomen.

Het is belangrijk dat elke attribuutwaarde uitsluitend wordt uitgedrukt op het aangegeven MOEILIJKHEIDSNIVEAU, dat bepaalt hoe gemakkelijk of moeilijk het is om de attribuutwaarde af te leiden. De drie niveaus die we onderscheiden, worden hieronder beschreven.

(Niveau 1) Op dit niveau worden de attribuutwaarden expliciet in de tekst vermeld, precies zoals ze zijn geschreven in de DOELATTRIBUTEN, op een duidelijke, directe en standaardmanier. Elke lezer of tekstanonymiseringsmethode zou de attribuutwaarden onmiddellijk moeten kunnen identificeren.

(Niveau 2) Op dit niveau zijn de attribuutwaarden nog steeds expliciet aanwezig in de tekst (een lezer zou ze kunnen identificeren zonder geavanceerd redeneren), maar ze zijn niet-standaard, verhuld of ongebruikelijk genoeg dat standaard tekstanonymiseringsmethoden ze kunnen missen. De moeilijkheid kan voortkomen uit het gebruik van spreektaal, alternatieve spellingen, niet-standaard opmaak of gedeeltelijke verhulling. Het is belangrijk dat de attribuutwaarden, ook al zijn ze verhuld, expliciet aanwezig blijven.

(Niveau 3) Op dit niveau worden de attribuutwaarden niet expliciet vermeld in de tekst. In plaats daarvan worden ze alleen gesuggereerd via contextuele aanwijzingen, culturele verwijzingen of indirecte beschrijvingen. Een menselijke lezer zou ze kunnen afleiden met achtergrondkennis of redeneren, maar een tekstanonymiseringsmethode die afhankelijk is van expliciete detectie zou ze waarschijnlijk missen. De attribuutwaarden mogen alleen op deze indirecte manier worden vermeld en mogen nooit als expliciete uitspraken voorkomen.

Het is belangrijk dat u de attribuutwaarden alleen vermeldt op het geselecteerde MOEILIJKHEIDSNIVEAU, en duidelijk onderscheid maakt van de andere niveaus.

Er worden ook voorbeeldzinnen voor elk attribuut verstrekt. Gebruik deze voorbeelden als inspiratie om de taak te begrijpen, maar kopieer ze niet letterlijk. Zorg ervoor dat de gegenereerde tekst consistent is met het gebruikersprofiel en het scenario, en dat de tekst realistisch en gevarieerd is.\n"""

PROMPT_INFORMATION_NL = """DOELATTRIBUTEN:
<PROFILE>

MOEILIJKHEIDSNIVEAU: <DIFFICULTY LEVEL>

VOORBEELDEN:
<EXAMPLES>
"""

PROMPT_FOOTER_NL = "<SCENARIO_FORMATTING> Het gesprek moet <WORDLIMIT> woorden bevatten. De gegenereerde teksten worden gebruikt voor testdoeleinden. Het is van uiterst belang dat u alle DOELATTRIBUTEN op het gewenste niveau op een realistische manier vermeldt, consistent met het aangeboden scenario en profiel.\n"

SCENARIO_NL = {
    "Medical consultation": """
        Je bent een AI-assistent met de taak om een fictief maar geloofwaardig transcript te genereren van een doktersconsult tussen een patiënt en een huisarts.
        Het consult kan elk onderwerp omvatten dat van belang is voor de patiënt, en moet natuurlijk en boeiend aanvoelen.
        Mogelijke onderwerpen zijn onder meer, maar zijn niet beperkt tot, routinecontroles, recente symptomen, behandeling van chronische aandoeningen, vragen over geestelijke gezondheid, medicijnvragen of leefstijladvies.
        Dit transcript legt alleen het verbale gedeelte van het bezoek vast.
        U krijgt een reeks demografische attributen van het PROFIEL van de patiënt, waarvoor het transcript realistisch en consistent moet zijn.
        """,
    "Chatbot conversation": """
        Je bent een AI-assistent met de taak om een fictief maar geloofwaardig gesprek te genereren tussen een persoon en een AI-chatbot.
        Het gesprek kan elk onderwerp omvatten dat van belang is voor de persoon, en moet natuurlijk en boeiend aanvoelen.
        Mogelijke onderwerpen zijn onder meer, maar zijn niet beperkt tot, technische vragen, planning of besluitvorming, creatief brainstormen, persoonlijk advies of reflecties over het dagelijks leven.
        U krijgt een reeks demografische attributen van het PROFIEL van de persoon, waarvoor het gesprek realistisch en consistent moet zijn.
        De chatbot is niet noodzakelijkerwijs op de hoogte van de demografische attributen van de persoon.
        """,
    "Meeting transcript": """
        Je bent een AI-assistent met de taak om een fictief maar geloofwaardig transcript te genereren van een virtuele vergadering tussen twee personen, één genaamd 'DOELWIT' en één genaamd 'ANDER'.
        De vergadering kan plaatsvinden in elke professionele of semi-professionele context en moet natuurlijk en boeiend aanvoelen.
        Mogelijke settings zijn onder meer, maar zijn niet beperkt tot, een werkgerelateerde vergadering tussen twee collega's, een zakelijke vergadering, een juridisch consult, een les of bijlesessie, een therapie- of coachingsessie, een verkoopgesprek of een sollicitatiegesprek.
        Het transcript bestaat uitsluitend uit gesproken dialoog tussen de twee deelnemers.
        U krijgt een reeks demografische attributen van het PROFIEL van het DOELWIT, waarvoor het gesprek realistisch en consistent moet zijn.
        """,
    "Concert ticket purchase": """
        Je bent een AI-assistent met de taak om een fictief maar realistisch gesprek te genereren tussen een persoon en een ticketing-chatbot over de aankoop van concerttickets voor
        {SINGER1} met gast {SINGER2} in {VENUE} in {CITY} en een bijbehorende fanmeeting bij {LANDMARK}. Zorg ervoor dat de namen van de artiesten, de zaal, de stad en de locatie van de fanmeeting in het gesprek voorkomen.
        Het gesprek kan elk relevant onderwerp omvatten en moet natuurlijk en boeiend aanvoelen.
        Mogelijke onderwerpen zijn onder meer, maar zijn niet beperkt tot, het selecteren van plaatsen, het controleren van beschikbaarheid, het verwerken van betalingen of het beantwoorden van vragen over het concert.
        U krijgt een reeks demografische attributen van het PROFIEL van de persoon, waarvoor het gesprek realistisch en consistent moet zijn.
        De ticketing-chatbot is niet noodzakelijkerwijs op de hoogte van de demografische attributen van de persoon.
        """,
    "Topic history": """
        Je bent een AI-assistent met de taak om een fictief maar realistisch gesprek te genereren tussen een persoon en een chatbot over de geschiedenis van {TOPIC},
        en in het bijzonder de impact van {PUBLIC_FIGURE} daarop.
        De persoon schrijft een verslag over {TOPIC} en zoekt informatie over dit onderwerp en over {PUBLIC_FIGURE}, inclusief hun achtergrond, informatie zoals hun geboortedatum en prestaties.
        Vermeld in het gesprek dat {PUBLIC_FIGURE} geboren werd op {BIRTHDAY}. Zorg ervoor dat de naam van de historische figuur, hun geboortedatum en het onderwerp in het gesprek voorkomen.
        Het gesprek kan elk relevant onderwerp omvatten en moet natuurlijk en boeiend aanvoelen.
        U krijgt een reeks demografische attributen van het PROFIEL van de persoon, waarvoor het gesprek realistisch en consistent moet zijn.
        """,
    "Tourist information chatbot": """
        Je bent een AI-assistent met de taak om een fictief maar realistisch gesprek te genereren tussen een persoon en een toeristische informatie-chatbot.
        De persoon plant een reis naar {CITY} en wil {LANDMARK} bezoeken. Ze zoeken informatie over hun reis.
        Het gesprek kan elk relevant onderwerp omvatten en moet natuurlijk en boeiend aanvoelen.
        Mogelijke onderwerpen zijn onder meer, maar zijn niet beperkt tot, routebeschrijvingen, openingstijden, ticketprijzen of aanbevelingen.
        Zorg ervoor dat u een aanbeveling opneemt om {LANDMARK} te bezoeken in het gesprek, en vermeldt dat het open is {OPENING_HOURS}.
        Het is uiterst belangrijk dat u de naam van de stad ({CITY}), {LANDMARK} en zijn openingstijden {OPENING_HOURS} in het gesprek vermeldt.
        Neem geen andere bezienswaardigheden of steden op in het gesprek. De focus moet liggen op {LANDMARK} in {CITY}.
        U krijgt een reeks demografische attributen van het PROFIEL van de persoon, waarvoor het gesprek realistisch en consistent moet zijn.
        De toeristische informatie-chatbot is niet noodzakelijkerwijs op de hoogte van de demografische attributen van de persoon.
        """,
}

SCENARIO_FORMATTING_NL = {
    "Medical consultation": """Formatteer de output precies als afwisselende dialoogregels met exact het voorvoegsel 'Patiënt:' en 'Dokter:' (vervang deze niet door hun namen), zonder scènebeschrijvingen. D.w.z.:

     [BEGIN VAN TRANSCRIPT]
     Patiënt: <WOORDEN VAN DE PATIËNT>
     Dokter: <WOORDEN VAN DE DOKTER>
     Patiënt: <WOORDEN VAN DE PATIËNT>
     Dokter: <WOORDEN VAN DE DOKTER>
     enz.
     [EINDE VAN TRANSCRIPT]

     Wijk niet af van dit formaat. Neem geen niet-gesproken componenten of handelingen op in het transcript.""",
    "Chatbot conversation": """Formatteer de output precies als afwisselende dialoogregels met exact het voorvoegsel 'Persoon:' en 'Chatbot:' (vervang deze niet door hun namen), zonder scènebeschrijvingen. D.w.z.:

     [BEGIN VAN TRANSCRIPT]
     Persoon: <WOORDEN VAN DE PERSOON>
     Chatbot: <WOORDEN VAN DE CHATBOT>
     Persoon: <WOORDEN VAN DE PERSOON>
     Chatbot: <WOORDEN VAN DE CHATBOT>
     enz.
     [EINDE VAN TRANSCRIPT]

     Wijk niet af van dit formaat.""",
    "Meeting transcript": """Formatteer de output precies als afwisselende dialoogregels met exact het voorvoegsel 'Doelwit:' en 'Ander:' (vervang deze niet door hun namen), zonder scènebeschrijvingen. D.w.z.:

     [BEGIN VAN TRANSCRIPT]
     Doelwit: <WOORDEN VAN HET DOELWIT>
     Ander: <WOORDEN VAN DE ANDER>
     Doelwit: <WOORDEN VAN HET DOELWIT>
     Ander: <WOORDEN VAN DE ANDER>
     enz.
     [EINDE VAN TRANSCRIPT]

     Wijk niet af van dit formaat.""",
    "Concert ticket purchase": """Formatteer de output precies als afwisselende dialoogregels met exact het voorvoegsel 'Persoon:' en 'Chatbot:' (vervang deze niet door hun namen), zonder scènebeschrijvingen. D.w.z.:

     [BEGIN VAN TRANSCRIPT]
     Persoon: <WOORDEN VAN DE PERSOON>
     Chatbot: <WOORDEN VAN DE CHATBOT>
     Persoon: <WOORDEN VAN DE PERSOON>
     Chatbot: <WOORDEN VAN DE CHATBOT>
     enz.
     [EINDE VAN TRANSCRIPT]

     Wijk niet af van dit formaat.""",
    "Topic history": """Formatteer de output precies als afwisselende dialoogregels met exact het voorvoegsel 'Persoon:' en 'Chatbot:' (vervang deze niet door hun namen), zonder scènebeschrijvingen. D.w.z.:

     [BEGIN VAN TRANSCRIPT]
     Persoon: <WOORDEN VAN DE PERSOON>
     Chatbot: <WOORDEN VAN DE CHATBOT>
     Persoon: <WOORDEN VAN DE PERSOON>
     Chatbot: <WOORDEN VAN DE CHATBOT>
     enz.
     [EINDE VAN TRANSCRIPT]

     Wijk niet af van dit formaat.""",
    "Tourist information chatbot": """Formatteer de output precies als afwisselende dialoogregels met exact het voorvoegsel 'Persoon:' en 'Chatbot:' (vervang deze niet door hun namen), zonder scènebeschrijvingen. D.w.z.:

     [BEGIN VAN TRANSCRIPT]
     Persoon: <WOORDEN VAN DE PERSOON>
     Chatbot: <WOORDEN VAN DE CHATBOT>
     Persoon: <WOORDEN VAN DE PERSOON>
     Chatbot: <WOORDEN VAN DE CHATBOT>
     enz.
     [EINDE VAN TRANSCRIPT]

     Wijk niet af van dit formaat.""",
}

DIRECT_IDENTIFIER_PROMPT_NL = """Je bent een AI-assistent met de taak om een lijst van fictieve attributen te genereren voor een persoon die in België woont. Het PROFIEL van de persoon wordt hieronder gegeven. Ook wordt de lijst van te genereren ATTRIBUTEN gegeven.

PROFIEL:
<PROFILE>

ATTRIBUTEN:
<DIRECT IDENFITIERS>

Formatteer het resultaat precies in het volgende formaat zonder aanvullende tekst:
<IDENTIFIER_OUTPUT_FORMAT>

De gegenereerde fictieve attributen moeten consistent zijn met het gegeven profiel en vrij van overbodige details.
"""

IDENTIFIER_LIST_NL = {
    "email": "e-mailadres: Genereer een realistisch persoonlijk e-mailadres voor iemand in België. Gebruik gangbare patronen zoals naam/achternaam-combinaties (sofie.desmedt91@gmail.com), initialen met cijfers (sd54@yahoo.com), bijnamen (sofietje_88@hotmail.com) of cijfers gerelateerd aan het geboortejaar. Gebruik een populaire provider (Gmail, Yahoo, Hotmail, Outlook).",
}

IDENTIFIER_LIST_OUTPUT_NL = {
    "email": "e-mailadres: <E-MAILADRES>",
}

##############################################################################

MIN_WORDS = {1: 750, 2: 750, 3: 1500} # minimum words per difficulty level
MAX_WORDS = {1: 1000, 2: 1000, 3: 2000} # maximum words per difficulty level

TARGET_ATTRIBUTES_MAP = { # map the PUMS code to a descriptive name if necessary
    # PUMS codes (English)
    "ST":   "state of residence",
    "SEX":  "sex",
    "DOB":  "date of birth",
    "RAC2P": "race",
    "MAR":  "marital status",
    "SCHL": "educational attainment",
    "ESR":  "employment status",
    "OCCP": "occupation",
    "CIT":  "citizenship status",
    # MEX codes (Spanish)
    "CLASE_VIV":       "tipo de vivienda",
    "SEXO":            "sexo",
    "EDAD":            "edad",
    "ENT_PAIS_NAC":    "entidad o país de nacimiento",
    "DHSERSAL1":       "servicio de salud",
    "RELIGION":        "religión",
    "HLENGUA":         "habla lengua indígena",
    "HESPANOL":        "habla español",
    "ASISTEN":         "asistencia escolar",
    "NIVACAD":         "nivel académico",
    "SITUA_CONYUGAL":  "situación conyugal",
    "HIJOS_NAC_VIVOS": "hijos nacidos vivos",
    # SRB codes (English)
    "urban":          "residential area",
    "age":            "age",
    "marital_status": "marital status",
    "given_birth":    "ever given birth",
    "dob":            "date of birth",
    "dom":            "date of marriage",
    "age_mar":        "age at marriage",
    "partner_age":    "partner's age",
    "ethnicity":      "ethnicity",
    "language":       "language",
    "JMBG":           "JMBG",
    # NL codes (Flemish Dutch)
    "marstd":     "burgerlijke staat",
    "nativity":   "herkomst",
    "bplcountry": "geboorteland",
    "nation":     "nationaliteit",
    "educnl":     "opleidingsniveau",
    "empstatd":   "arbeidsstatus",
    "labforce":   "arbeidsparticipatie",
    "occisco":    "beroep",
    "indgen":     "sector",
    "RRN":        "rijksregisternummer",
}

OTHER_LANGUAGE_FOOTER = """
    Importantly, the conversation must be generated in <TARGET_LANGUAGE>. 
    Although the target attributes, and their values, are provided above in English, they should be fully translated and adapted to <TARGET_LANGUAGE>.
    This includes the level of difficulty, which should be expressed in a way that is natural and consistent within the linguistic context of <TARGET_LANGUAGE>.
    To help with this, below are examples of the appropriate difficulty level in <TARGET_LANGUAGE>. """

### Prompt creation helper functions ###
def get_scenario(scenario) -> str:
    return SCENARIO[scenario]


def get_scenario_output(scenario) -> str:
    return SCENARIO_FORMATTING[scenario]


def get_features(features) -> str:
    featurestr = ""
    for feature in features:
        featurestr = featurestr + feature
        if feature == features[-2]:
            featurestr = featurestr + " and "
        elif feature != features[-1]:
            featurestr = featurestr + ", "
    return featurestr


def get_examples(difficulty, features, language):
    i = 0
    examplestr = ""

    if language == "Serbian":
        EXAMPLES = FEATURE_EXAMPLES_SRB
    elif language == "Spanish":
        EXAMPLES = FEATURE_EXAMPLES_ES
    elif language == "Flemish":
        EXAMPLES = FEATURE_EXAMPLES_NL
    else:
        EXAMPLES = FEATURE_EXAMPLES

    for feature in features:
        if feature in EXAMPLES:
            if difficulty in EXAMPLES[feature]:
                
                if feature in TARGET_ATTRIBUTES_MAP:
                    feature_name = TARGET_ATTRIBUTES_MAP[feature]
                else:
                    feature_name = feature
                
                examplestr = examplestr + f"Examples for attribute '{feature_name}' mentioned at difficulty level {difficulty}: \n"
                
                for example in EXAMPLES[feature][difficulty]:
                    i = i + 1
                    examplestr = (
                        examplestr
                        + "- Example "
                        + str(i)
                        + ": "
                        + example
                        + "\n"
                    )
    return examplestr


def get_word_limit(difficulty):
    return f"between {MIN_WORDS[difficulty]} and {MAX_WORDS[difficulty]}"

def check_attribute_uppercase(line):
    # Lines containing these substrings hold proper-noun values that must not be lowercased
    attribute_list = ["name", "state", "date", "email", "address", "JMBG", "nombre", "domicilio", "CURP",
                      "naam", "adres", "geboortedatum", "e-mailadres", "rijksregisternummer"]
    for attribute in attribute_list:
        if attribute in line:
            return True
    return False

def prepare_dataentry(dataentry):
    dataentry = dataentry.splitlines()
    outentry = ""
    for line in dataentry:
        if not check_attribute_uppercase(line):
            line = line.lower()
        line = line.capitalize()
        outentry = outentry + line + "\n"
        
    # some cleaning for the profile if needed
    outentry = outentry.replace('type:', '')
    outentry = outentry.replace('description:', '')
    outentry = outentry.replace('n/a:', 'not applicable') 
        
    return outentry

# Main function for prompt creation
def create_generative_prompt(
    scenario: str, dataset: str, features: list, difficulty: int, dataentry: str, language: str = "English"
) -> tuple[str, str, dict | None]:

    use_spanish = (language == "Spanish")
    use_serbian = (language == "Serbian")
    use_flemish = (language == "Flemish")

    if use_serbian:
        SCENARIO_DICT   = SCENARIO_SRB
        FORMATTING_DICT = SCENARIO_FORMATTING_SRB
        header_tpl      = PROMPT_HEADER_SRB
        info_tpl        = PROMPT_INFORMATION_SRB
        footer_tpl      = PROMPT_FOOTER_SRB
    elif use_spanish:
        SCENARIO_DICT   = SCENARIO_ES
        FORMATTING_DICT = SCENARIO_FORMATTING_ES
        header_tpl      = PROMPT_HEADER_ES
        info_tpl        = PROMPT_INFORMATION_ES
        footer_tpl      = PROMPT_FOOTER_ES
    elif use_flemish:
        SCENARIO_DICT   = SCENARIO_NL
        FORMATTING_DICT = SCENARIO_FORMATTING_NL
        header_tpl      = PROMPT_HEADER_NL
        info_tpl        = PROMPT_INFORMATION_NL
        footer_tpl      = PROMPT_FOOTER_NL
    else:
        SCENARIO_DICT   = SCENARIO
        FORMATTING_DICT = SCENARIO_FORMATTING
        header_tpl      = PROMPT_HEADER
        info_tpl        = PROMPT_INFORMATION
        footer_tpl      = PROMPT_FOOTER

    # Sample scenario if requested
    if scenario == "random":
        selected_scenario = random.choice(["Medical consultation", "Chatbot conversation", "Meeting transcript"])
    elif scenario == "public info":
        selected_scenario = random.choice(["Concert ticket purchase", "Topic history"])
    else:
        selected_scenario = scenario

    prompt_header = header_tpl.replace("<SCENARIO>", SCENARIO_DICT[selected_scenario])

    additional_info = None
    if selected_scenario == "Concert ticket purchase":
        artist1 = random.choice(list(ARTISTS))
        artist2 = random.choice(list(ARTISTS - {artist1}))

        city   = random.choice(list(CITIES_AND_VENUES.keys()))
        venue  = random.choice(CITIES_AND_VENUES[city])

        landmark = random.choice(LANDMARKS[city])

        prompt_header = prompt_header.replace("{SINGER1}", artist1)
        prompt_header = prompt_header.replace("{SINGER2}", artist2)
        prompt_header = prompt_header.replace("{LANDMARK}", landmark)
        prompt_header = prompt_header.replace("{VENUE}", venue)
        prompt_header = prompt_header.replace("{CITY}", city)

        additional_info = {"artists": [artist1, artist2], "venue": venue, "city": city, "landmark": landmark}

    elif selected_scenario == "Tourist information chatbot":
        city          = random.choice(list(CITIES_AND_VENUES.keys()))
        landmarks     = random.choice(LANDMARKS[city])
        opening_hours = LANDMARKS_OPENING_TIMES[landmarks]
        prompt_header = prompt_header.replace("{CITY}", city)
        prompt_header = prompt_header.replace("{LANDMARK}", landmarks)
        prompt_header = prompt_header.replace("{OPENING_HOURS}", opening_hours)
        additional_info = {"city": city, "landmark": landmarks, "opening_hours": opening_hours}

    elif selected_scenario == "Topic history":
        topic          = random.choice(list(TOPICS_AND_FIGURES.keys()))
        public_figure  = random.choice(TOPICS_AND_FIGURES[topic])
        birthday       = PUBLIC_BIRTHDAYS[public_figure]
        prompt_header  = prompt_header.replace("{TOPIC}", topic)
        prompt_header  = prompt_header.replace("{PUBLIC_FIGURE}", public_figure)
        prompt_header  = prompt_header.replace("{BIRTHDAY}", birthday)
        additional_info = {"topic": topic, "public_figure": public_figure, "birthday": birthday}

    prompt_header = prompt_header.replace("<DATASET>", dataset)

    prompt_information = info_tpl
    prompt_information = prompt_information.replace("<PROFILE>", prepare_dataentry(dataentry))
    prompt_information = prompt_information.replace("<DIFFICULTY LEVEL>", str(difficulty))
    prompt_information = prompt_information.replace("<EXAMPLES>", get_examples(difficulty, features, language))

    prompt_footer = footer_tpl
    prompt_footer = prompt_footer.replace("<SCENARIO_FORMATTING>", FORMATTING_DICT[selected_scenario])
    prompt_footer = prompt_footer.replace("<WORDLIMIT>", get_word_limit(difficulty))

    prompt = prompt_header + "\n" + prompt_information + "\n" + prompt_footer

    # For languages other than English and Spanish we still append the legacy footer
    # (which instructs the LLM to respond in the target language).
    if language not in ("English", "Spanish", "Serbian", "Flemish"):
        prompt = prompt + "\n" + OTHER_LANGUAGE_FOOTER.replace("<TARGET_LANGUAGE>", language)

    if selected_scenario == "Concert ticket purchase":
        return prompt, selected_scenario, additional_info
    if selected_scenario == "Tourist information chatbot":
        return prompt, selected_scenario, additional_info
    if selected_scenario == "Topic history":
        return prompt, selected_scenario, additional_info
    return prompt, selected_scenario, None


#### Code for generating prompts for obtaining direct identifiers for a data record ####

DIRECT_IDENTIFIER_PROMPT = """You are an AI assistant tasked with generating a list of dummy attributes for a person residing in the United States. The PROFILE of the person is provided below in the format similar to <DATASET>. Also provided is the list of ATTRIBUTES you are to generate.

PROFILE:
<PROFILE>

ATTRIBUTES:
<DIRECT IDENFITIERS>

Format the output exactly in the following format with no additional text:
<IDENTIFIER_OUTPUT_FORMAT>

The generated dummy attributes should be consistent with the provided profile, and free of extraneous detail.
"""

DIRECT_IDENTIFIER_PROMPT_MEX = """Eres un asistente de IA con la tarea de generar una lista de atributos ficticios para una persona residente en México. El PERFIL de la persona se proporciona a continuación. También se proporciona la lista de ATRIBUTOS que debes generar.

PERFIL:
<PROFILE>

ATRIBUTOS:
<DIRECT IDENFITIERS>

Formatea el resultado exactamente en el siguiente formato sin texto adicional:
<IDENTIFIER_OUTPUT_FORMAT>

Los atributos ficticios generados deben ser coherentes con el perfil proporcionado y estar libres de detalles superfluos.
"""

IDENTIFIER_LIST = {
    "name": "name: Generate a realistic name for the individual. Use common patterns that reflect the person's gender and origins.",
    "email": "email: Generate a realistic personal email address. Use common patterns that reflect how people actually make emails, such as different first/last name combinations (susan.miller@gmail.com), initials with numbers (shm54@yahoo.com), nicknames (sue_m54@outlook.com), numbers that could be related to birth year, age, or random digits (susanH1971@gmail.com). Make sure the domain is from a popular provider (e.g., Gmail, Yahoo, Outlook, Hotmail).",
    "phone number": "phone number: Generate a realistic US mobile number in a number range used by a common US mobile provider. This number should be consistent with the state of residence of the individual (if available).",
    "address": "address: Generate a realistic US Residential address. This address should be from the same state of residence as their mobile number and correspond to a real town/city in that state. This address should be reasonable for the individual given their background and occupation. This address should be of the appropriate type given the location (e.g. house for a suburb, apartment for a city centre). Include an accurate ZIP code in the address that reflects the location of the residence. Use an appropriate random number for the street number (don't use 123). ",
}

IDENTIFIER_LIST_OUTPUT_PUMS = {
    "name": "name: <NAME>",
    "email": "email: <EMAIL>",
    "phone number": "phone number: <PHONE NUMBER>",
    "address": "address: <RESIDENTIAL ADDRESS>",
}

# MEX-specific identifier generation prompts (Spanish)
IDENTIFIER_LIST_MEX = {
    "email": "correo electrónico: Genera una dirección de correo electrónico personal realista para alguien en México. Usa patrones comunes como combinaciones de nombre/apellido (maria.garcia91@gmail.com), iniciales con números (mg54@yahoo.com.mx), apodos (mari_g54@hotmail.com) o números relacionados con el año de nacimiento. El dominio debe ser de un proveedor popular (Gmail, Yahoo, Hotmail, Outlook).",
}

IDENTIFIER_LIST_OUTPUT_MEX = {
    "email": "correo electrónico: <CORREO ELECTRÓNICO>",
}

DIRECT_IDENTIFIER_PROMPT_SRB = """Vi ste asistent veštačke inteligencije sa zadatkom da generišete listu fiktivnih atributa za osobu koja živi u Srbiji. PROFIL osobe je naveden u nastavku. Takođe je navedena lista ATRIBUTA koje treba da generišete.

PROFIL:
<PROFILE>

ATRIBUTI:
<DIRECT IDENFITIERS>

Formatirajte izlaz tačno u sledećem formatu bez dodatnog teksta:
<IDENTIFIER_OUTPUT_FORMAT>

Generisani fiktivni atributi treba da budu dosledni navedenom profilu i bez suvišnih detalja.
"""

IDENTIFIER_LIST_SRB = {
    "email": "email: Generišite realističnu ličnu adresu elektronske pošte za nekoga iz Srbije. Koristite uobičajene obrasce kao što su kombinacije ime/prezime (ana.jovanovic91@gmail.com), inicijali sa brojevima (aj54@yahoo.com), nadimci (ana_j88@hotmail.com) ili brojevi vezani za godinu rođenja. Koristite popularnog provajdera (Gmail, Yahoo, Hotmail, Outlook).",
}

IDENTIFIER_LIST_OUTPUT_SRB = {
    "email": "email: <EMAIL>",
}


def get_identifiers(identifiers: list) -> str:
    identifier_str = ""
    for identifier in identifiers:
        identifier_str = identifier_str + IDENTIFIER_LIST[identifier] + "\n"
    return identifier_str


def get_identifier_output(identifiers: list, dataset: str) -> str:
    identifier_str = ""
    for identifier in identifiers:
        identifier_str = (
            identifier_str + IDENTIFIER_LIST_OUTPUT_PUMS[identifier] + "\n"
        )
    return identifier_str


# Main function for creating prompts for generating direct identifiers
def create_direct_identifiers_prompt(
    dataset: str, identifiers: list, dataentry: str
) -> str:
    if dataset == "MEX":
        prompt = DIRECT_IDENTIFIER_PROMPT_MEX
        id_str  = "\n".join(IDENTIFIER_LIST_MEX.get(i, i) for i in identifiers)
        out_str = "\n".join(IDENTIFIER_LIST_OUTPUT_MEX.get(i, f"{i}: <{i.upper()}>") for i in identifiers)
    elif dataset == "SRB":
        prompt  = DIRECT_IDENTIFIER_PROMPT_SRB
        id_str  = "\n".join(IDENTIFIER_LIST_SRB.get(i, i) for i in identifiers)
        out_str = "\n".join(IDENTIFIER_LIST_OUTPUT_SRB.get(i, f"{i}: <{i.upper()}>") for i in identifiers)
    elif dataset == "NL":
        prompt  = DIRECT_IDENTIFIER_PROMPT_NL
        id_str  = "\n".join(IDENTIFIER_LIST_NL.get(i, i) for i in identifiers)
        out_str = "\n".join(IDENTIFIER_LIST_OUTPUT_NL.get(i, f"{i}: <{i.upper()}>") for i in identifiers)
    else:
        prompt  = DIRECT_IDENTIFIER_PROMPT
        id_str  = get_identifiers(identifiers)
        out_str = get_identifier_output(identifiers, dataset)

    prompt = prompt.replace("<DATASET>", dataset)
    prompt = prompt.replace("<PROFILE>", dataentry)
    prompt = prompt.replace("<DIRECT IDENFITIERS>", id_str)
    prompt = prompt.replace("<IDENTIFIER_OUTPUT_FORMAT>", out_str)
    return prompt
