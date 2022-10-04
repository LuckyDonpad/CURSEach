# класс среды обитания особей хранящий в себе:
# train_x - тренировочный набор данных
# train_y - набор ответов к тренировочному набору данных
# val_x - валидационный набор данных
# val_y - набор ответов к валидационному набору данных
class Environment(object):
    def __init__(self, train_x, train_y, val_x, val_y):
        self.train_x = train_x
        self.train_y = train_y
        self.val_y = val_y
        self.val_x = val_x