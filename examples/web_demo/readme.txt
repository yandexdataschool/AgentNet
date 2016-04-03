This is a minimalistic demo of user-agent interaction within a wikicat problem.

It requires flask and can be run as 
python app.py

The description of UI is only given in russian at this point, sorry.

[панелька слева]
 
Первое поле - путь до файла с состоянием.
 
Она обновляется сама, но можно скопипастить себе состояние и потом к нему вернуться, просто вставив в это поле.
 
Второе поле - ответ на вопрос агента - тут может быть 0, 1 или пустая строка.
0 или 1 - ответ "да" или "нет"
пустая строка (или любое не-число) - показывает состояние из первого поля + действие агента.
 
Кнопки
Update - ответить и получить следующий вопрос
New session - начать новую сессию
 
Типичный паттерн взаимодействия -
New session -> ответ во второе поле -> update -> ответ во второе поле -> update -> ...
 
Формат ответа агента:
 
Во втором поле 0 или 1
New state:
./states/state21 // новое состояние (также обновилось в панели слева)

Agent action: decades_active:2010 //действие агента - это вопрос или действие end_session_now

Top-5 Qvalues: //наибольшие 5 Q-значений для value-based и топ-5 вероятностей действий для policy-based
decades_active:2010 : 4.04129
Instruments:guitar : 3.35747
Occupation:songwriter : 3.31371
Genres:pop : 3.28995
decades_active:1990 : 3.27812
 
Во втором окне пустая/другая строка
Could not read response. Has to be 1 or 0 in this setup.
Showing state instead. // показана информация о текущем, а не следующем состоянии
 

Agent action: last_activity:still_active //последний вопрос агента

Top-5 Qvalues: //топ-5 Qзначений
last_activity:still_active : 4.24501
Instruments:guitar : 3.30388
decades_active:2000 : 3.20795
Occupation:songwriter : 3.08942
Genres:rock : 3.07411
