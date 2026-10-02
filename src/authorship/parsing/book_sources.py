"""URL lists for the book corpus.

Training and validation sets are kept separate for easier dataset
management. Comments identify the author and title behind each URL.
"""

TRAINING_URLS: list[str] = [
    # Іван Нечуй-Левицький,
    # "Хмари"
    "https://www.ukrlib.com.ua/books/printitzip.php?tid=910",
    # "Гетьман Іван Виговський"
    "https://www.ukrlib.com.ua/books/printitzip.php?tid=902",
    # "Дві московки"
    "https://www.ukrlib.com.ua/books/printitzip.php?tid=904",
    # Панас Мирний,
    # "Повія"
    "https://www.ukrlib.com.ua/books/printitzip.php?tid=982",
    # "Хіба ревуть воли, як ясла повні?"
    "https://www.ukrlib.com.ua/books/printitzip.php?tid=983",
    # Іван Котляревський, "Енеїда" (поламане)
    "https://www.ukrlib.com.ua/books/printitzip.php?tid=1052",
    # Валер'ян Підмогильний,
    # "Місто"
    "https://www.ukrlib.com.ua/books/printitzip.php?tid=76",
    # "В епідемічному бараці"
    "https://www.ukrlib.com.ua/books/printitzip.php?tid=1919",
    # "Добрий Бог"
    "https://www.ukrlib.com.ua/books/printitzip.php?tid=1926",
    # "Військовий літун"
    "https://www.ukrlib.com.ua/books/printitzip.php?tid=1914",
    # "Третя революція"
    "https://www.ukrlib.com.ua/books/printitzip.php?tid=1912",
    # "Повість без назви"
    "https://www.ukrlib.com.ua/books/printitzip.php?tid=1910",
    # "На селі"
    "https://www.ukrlib.com.ua/books/printitzip.php?tid=1920",
    # Пантелеймон Куліш,
    # "Чорна рада"
    "https://www.ukrlib.com.ua/books/printitzip.php?tid=1084",
    # "Байда князь Вишневецький" (поламане)
    "https://www.ukrlib.com.ua/books/printitzip.php?tid=2602",
    # "Маруся Богуславка" (поламане)
    "https://www.ukrlib.com.ua/books/printitzip.php?tid=1077",
    # Ольга Кобилянська,
    # "Земля"
    "https://www.ukrlib.com.ua/books/printitzip.php?tid=1030",
    # "Меланхолійний Вальс"
    "https://www.ukrlib.com.ua/books/printitzip.php?tid=2457",
    # "Вовчиха"
    "https://www.ukrlib.com.ua/books/printitzip.php?tid=2460",
    # "Царівна"
    "https://www.ukrlib.com.ua/books/printitzip.php?tid=1027",
    # "Юда"
    "https://www.ukrlib.com.ua/books/printitzip.php?tid=2471",
    # "Природа"
    "https://www.ukrlib.com.ua/books/printitzip.php?tid=2470",
    # Микола Гоголь,
    # "Тарас Бульба"
    "https://www.ukrlib.com.ua/world/printitzip.php?tid=15",
    # "Вечори на хуторі біля Диканьки" (збірка)
    "https://www.ukrlib.com.ua/world/printitzip.php?tid=4",
    # "Ревізор"
    "https://www.ukrlib.com.ua/world/printitzip.php?tid=405",
    # "Шинель"
    "https://www.ukrlib.com.ua/world/printitzip.php?tid=402",
    # "Мертві душі"
    "https://www.ukrlib.com.ua/world/printitzip.php?tid=3",
    # Улас Самчук,
    # "Плянета Ді-Пі"
    "https://www.ukrlib.com.ua/books/printitzip.php?tid=3651",
    # "Дермань"
    "https://www.ukrlib.com.ua/books/printitzip.php?tid=27560",
    # "Віднайдений рай" (збірка)
    "https://www.ukrlib.com.ua/books/printitzip.php?tid=3647",
    # Іван Карпенко-Карий,
    # "Мартин Боруля"
    "https://www.ukrlib.com.ua/books/printitzip.php?tid=1009",
    # "Суєта"
    "https://www.ukrlib.com.ua/books/printitzip.php?tid=2345",
    # "Сава Чалий"
    "https://www.ukrlib.com.ua/books/printitzip.php?tid=1008",
    # "Сто тисяч"
    "https://www.ukrlib.com.ua/books/printitzip.php?tid=1007",
    # "Житейське море"
    "https://www.ukrlib.com.ua/books/printitzip.php?tid=2349",
    # "Наймичка"
    "https://www.ukrlib.com.ua/books/printitzip.php?tid=2348",
    # "Хазяїн"
    "https://www.ukrlib.com.ua/books/printitzip.php?tid=1006",
    # "Бондарівна"
    "https://www.ukrlib.com.ua/books/printitzip.php?tid=2350",
    # "Розумний і дурень"
    "https://www.ukrlib.com.ua/books/printitzip.php?tid=2346",
    # "Новобранець"
    "https://www.ukrlib.com.ua/books/printitzip.php?tid=2347",
    # "Невольник"
    "https://www.ukrlib.com.ua/books/printitzip.php?tid=1934",
    # "Бурлака"
    "https://www.ukrlib.com.ua/books/printitzip.php?tid=1010",
    # "Безталанна"
    "https://www.ukrlib.com.ua/books/printitzip.php?tid=1011",
    # Іван Франко,
    # "Захар Беркут"
    "https://www.ukrlib.com.ua/books/printitzip.php?tid=645",
    # "Борислав сміється"
    "https://www.ukrlib.com.ua/books/printitzip.php?tid=625",
    # "Батьківщина"
    "https://www.ukrlib.com.ua/books/printitzip.php?tid=3792",
    # "Коли ще звірі говорили" (збірка)
    "https://www.ukrlib.com.ua/books/printitzip.php?tid=9688",
    # "Основи суспільності"
    "https://www.ukrlib.com.ua/books/printitzip.php?tid=3816",
    # "Украдене щастя"
    "https://www.ukrlib.com.ua/books/printitzip.php?tid=653",
    # "Іван Вишневський" (поламане)
    # "https://www.ukrlib.com.ua/books/printitzip.php?tid=623",
    # Анатолій Шиян, "Партизанський край"
    "https://www.ukrlib.com.ua/books/printitzip.php?tid=8941",
    # Зарубіжна література
    # Джордж Орвелл,
    # "1984"
    "https://www.ukrlib.com.ua/world/printitzip.php?tid=2283",
    # "Скотоферма"
    "https://www.ukrlib.com.ua/world/printitzip.php?tid=2284",
    # Брати (Аркадій і Борис) Стругацькі,
    # "Понеділок починається в суботу"
    "https://www.ukrlib.com.ua/world/printitzip.php?tid=4072",
    # "Пікнік на узбіччі"
    "https://www.ukrlib.com.ua/world/printitzip.php?tid=4073",
    # "Хлопець із пекла"
    "https://www.ukrlib.com.ua/world/printitzip.php?tid=4074",
    # "Малюк"
    "https://www.ukrlib.com.ua/world/printitzip.php?tid=4069",
    # Михайло Коцюбинський,
    # "Тіні забутих предків"
    "https://www.ukrlib.com.ua/books/printitzip.php?tid=1058",
    # "Fata Morgana"
    "https://www.ukrlib.com.ua/books/printitzip.php?tid=1066",
    # "Intermezzo"
    "https://www.ukrlib.com.ua/books/printitzip.php?tid=1065",
    # "В путах шайтана"
    "https://www.ukrlib.com.ua/books/printitzip.php?tid=2539",
    # "Відьма"
    "https://www.ukrlib.com.ua/books/printitzip.php?tid=2561",
    # Леся Українка,
    # "Лісова пісня"
    "https://www.ukrlib.com.ua/books/printitzip.php?tid=508",
    # "Камінний господар"
    "https://www.ukrlib.com.ua/books/printitzip.php?tid=501",
    # "Бояриня"
    "https://www.ukrlib.com.ua/books/printitzip.php?tid=481",
    # "Кассандра"
    "https://www.ukrlib.com.ua/books/printitzip.php?tid=10382",
    # "В катакомбах"
    "https://www.ukrlib.com.ua/books/printitzip.php?tid=485",
    # Марко Вовчок,
    # "Інститутка"
    "https://www.ukrlib.com.ua/books/printitzip.php?tid=457",
    # "Народні оповідання" (збірка)
    "https://www.ukrlib.com.ua/books/printitzip.php?tid=6737",
    # Михайло Стельмах,
    # "Правда і Кривда"
    "https://www.ukrlib.com.ua/books/printitzip.php?tid=331",
    # "Чотири броди"
    "https://www.ukrlib.com.ua/books/printitzip.php?tid=332",
    # Олесь Гончар,
    # "Собор"
    "https://www.ukrlib.com.ua/books/printitzip.php?tid=550",
    # "Тронка"
    "https://www.ukrlib.com.ua/books/printitzip.php?tid=552",
    # Остап Вишня,
    # "Літературно-мистецьки усмішки" (збірка)
    "https://www.ukrlib.com.ua/books/printitzip.php?tid=3048",
    # "Мисливські усмішки"
    "https://www.ukrlib.com.ua/books/printitzip.php?tid=449",
    # Микола Хвильовий,
    # "Санаторійна зона"
    "https://www.ukrlib.com.ua/books/printitzip.php?tid=2736",
    # "Вальдшнепи"
    "https://www.ukrlib.com.ua/books/printitzip.php?tid=666",
    # Юрій Яновський,
    # "Вершники"
    "https://www.ukrlib.com.ua/books/printitzip.php?tid=53",
    # "Майстер Корабля"
    "https://www.ukrlib.com.ua/books/printitzip.php?tid=54",
    # Григір Тютюнник,
    # "Вир"
    "https://www.ukrlib.com.ua/books/printitzip.php?tid=464",
    # English-language books (already tested), currently commented out:
    # Arthur Conan Doyle
    # "https://www.gutenberg.org/cache/epub/2852/pg2852.txt",
    # Daniel Defoe
    # "https://www.gutenberg.org/cache/epub/521/pg521.txt",
    # Joseph Conrad
    # "https://www.gutenberg.org/cache/epub/219/pg219.txt",
    # Oscar Wilde
    # "https://www.gutenberg.org/cache/epub/174/pg174.txt",
    # Jonathan Swift
    # "https://www.gutenberg.org/cache/epub/1080/pg1080.txt",
]

VALIDATION_URLS: list[str] = [
    # Роман Федорів,
    # "У пущах карпатських"
    "https://www.ukrlib.com.ua/books/printitzip.php?tid=4758",
    # Юрій Косач,
    # "Еней та життя інших"
    "https://www.ukrlib.com.ua/books/printitzip.php?tid=4392",
    # Павло Загребельний,
    # "Південний комфорт"
    "https://www.ukrlib.com.ua/books/printitzip.php?tid=4344",
    # Володимир Винниченко,
    # "Брехня"
    "https://www.ukrlib.com.ua/books/printitzip.php?tid=4089",
    # Олександр Гаврош,
    # "Ромео і Жасмин"
    "https://www.ukrlib.com.ua/books/printitzip.php?tid=26975",
]
