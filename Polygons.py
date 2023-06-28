import math
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np


fig = plt.figure("Polygons", figsize=(10, 10))
dec = plt.axes(xlim = (-1.1, 1.1), ylim = (-1.1, 1.1))

P = int(input("Введите P, количество сторон многоугольника: "))
Q = int(input("Введите Q, количество многоугольников, cходящихся в каждой вершине: "))
N = int(input("Введите количество слоев: "))
ZERO_H = np.array([[0, 0, 1]])
4
colors = ["black", "blue", "red", "green", "orange"]

DEBUG_MODE = 0  # Режим отладки. В нем отображается построение каждого отдельного флага, отдельных слоев
# и отображаются общие вершины. Окна надо каждый раз закрывать, так как я пока что не умею делать анимацию


# RAD - это евклидово расстояние от центра до вершины, HL - до середины стороны.
ChRADl = (math.cos(math.pi / P) * math.cos(math.pi / Q)) / (math.sin(math.pi / P) * math.sin(math.pi / Q))
ChHl = math.cos(math.pi / Q) / math.sin(math.pi / P)
RAD = (math.exp(math.acosh(ChRADl)) - 1) / (1 + math.exp(math.acosh(ChRADl)))
HL = (math.exp(math.acosh(ChHl)) - 1) / (1 + math.exp(math.acosh(ChHl)))
MID = np.array([[HL * math.sin(math.pi / P), HL * math.cos(math.pi / P)]])



a = "a"
b = "b"
c = "ab"

# Логика и вычисления
def equal(a, b):
    for i in range(len(a[0])):
        if abs(a[0][i] - b[0][i]) > 0.000000001:
            return 0
    return 1


def sign(x): #знак числа
    if x >= 0:
        return 1
    else:
        return -1


def proj(t): #проекция круга на гиперболоид
    x_t = t[0][0]
    y_t = t[0][1]
    x_f = 2 * x_t / (1 - x_t ** 2 - y_t ** 2)
    y_f = 2 * y_t / (1 - x_t ** 2 - y_t ** 2)
    z_f = math.sqrt(x_f ** 2 + y_f ** 2 + 1)
    return np.array([[x_f, y_f, z_f]])


def proj_r(t): #пррекция гиперболоида в круг
    x_t = t[0][0]
    y_t = t[0][1]
    return np.array(
        [(x_t / (math.sqrt(x_t ** 2 + y_t ** 2 + 1) + 1), (y_t / (math.sqrt(x_t ** 2 + y_t ** 2 + 1) + 1)))])


def rotate_3d(fi):  #поворот вокруг Oz в R3 на угол fi
    c = math.cos(fi)
    s = math.sin(fi)
    return np.array([[c, s, 0],
                     [-s, c, 0],
                     [0, 0, 1]])


def ang(t): #аргумент точки t в полярной (цилиндрической) системе координат, от -π до π
    if t[0][0] ** 2 + t[0][1] ** 2 == 0:
        return 0
    if t[0][0] > 0:
        return math.asin(t[0][1] / math.sqrt(t[0][1] ** 2 + t[0][0] ** 2))
    else:
        return math.pi * sign(t[0][1]) - math.asin(t[0][1] / math.sqrt(t[0][1] ** 2 + t[0][0] ** 2))


def dist_angle(t, q):
    d = ang(t) - ang(q)
    if d > math.pi:
        d -= 2*math.pi
    if d < -math.pi:
        d += 2*math.pi
    return d


def clock_check(Polygon1, Polygon2): #Определяет, какой из многоугольников расположен дальше против часовой стрелки
    min = 4
    max = -4
    for i in range(P):
        for j in range(P):
            if dist_angle(Polygon1.pol[i].dot, Polygon2.pol[j].dot) < min:
                min = dist_angle(Polygon1.pol[i].dot, Polygon2.pol[j].dot)
            if dist_angle(Polygon1.pol[i].dot, Polygon2.pol[j].dot) > max:
                max = dist_angle(Polygon1.pol[i].dot, Polygon2.pol[j].dot)
    if abs(max) > abs(min):
        return 1
    elif abs(max) < abs(min):
        return -1
    else:
        return 0


def rotate_h(fi): #гиперболический сдвиг на аргумент fi
    c = math.cosh(fi)
    s = math.sinh(fi)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [s, 0, c]])


def shift_h(k, m): #преобразование на гиперболоиде, переводящее k в m
    M = rotate_3d(-ang(k))
    angm = ang(m)
    k = k @ rotate_3d(-ang(k))
    m = m @ rotate_3d(-angm)
    M = M @ rotate_h(math.asinh(m[0][0]) - math.asinh(k[0][0]))
    return M @ rotate_3d(angm)

# Рисование

def draw_line_arc(a, b, c="blue"):#рисование отрезка ab
    x0 = a[0][0]
    y0 = a[0][1]
    x1 = b[0][0]
    y1 = b[0][1]
    dist_mid = ((x1 + x0)**2 + (y1 + y0)**2)/4
    LW = 3/math.pow((1 + abs(math.log(1 - math.sqrt(dist_mid)))), 1)
    x2 = x1 / (x1 ** 2 + y1 ** 2)
    y2 = y1 / (x1 ** 2 + y1 ** 2)
    D = 2 * (x0 * (y1 - y2) + x1 * (y2 - y0) + x2 * (y0 - y1))#с википедии
    if abs(y0) < 0.0000000001 and abs(y1) < 0.0000000001:
        plt.plot([x0, x1], [y0, y1], color=c, linewidth=LW)
        return
    if abs(x0/y0 - x1/y1) < 0.000000001:
        plt.plot([x0, x1], [y0, y1], color=c, linewidth=LW)
        return
    Cx = ((x0 ** 2 + y0 ** 2) * (y1 - y2) + (x1 ** 2 + y1 ** 2) * (y2 - y0) + (x2 ** 2 + y2 ** 2) * (y0 - y1)) / D#с википедии
    Cy = ((x0 ** 2 + y0 ** 2) * (x2 - x1) + (x1 ** 2 + y1 ** 2) * (x0 - x2) + (x2 ** 2 + y2 ** 2) * (x1 - x0)) / D#с википедии
    C = np.array([[Cx, Cy]])
    R = math.sqrt((Cx - x0) ** 2 + (Cy - y0) ** 2)

    ang_arc = ang(b - C) - ang(a - C)
    #draw_circle(C, R)
    while ang_arc < -math.pi or ang_arc > math.pi:
        if ang_arc > math.pi:
            ang_arc -= 2 * math.pi
        else:
            ang_arc += 2 * math.pi

    arc_x = []
    arc_y = []
    M = 100
    for alpha in range(M + 1):
        arc_x += [Cx + R * math.cos(ang(a - C) + ang_arc*float(alpha) / float(M))]
        arc_y += [Cy + R * math.sin(ang(a - C) + ang_arc*float(alpha) / float(M))]
    plt.plot(arc_x, arc_y, color=c, linewidth=LW)#, linewidth=LW)
    #draw_dot(a)
    #draw_dot(b)


def draw_circle(C, R): #рисование окружности по центру и радиусу
    Cx = C[0][0]
    Cy = C[0][1]
    arc_x = []
    arc_y = []
    for cnt in range(0, int(2 * math.pi * 10000), 1):
        arc_x += [Cx + R * math.cos(float(cnt) / 10000)]
        arc_y += [Cy + R * math.sin(float(cnt) / 10000)]
    plt.plot(arc_x, arc_y)


def draw_dot(a, c = "black"):#рисование точки (точка - это вектор numpy)
    plt.scatter([a[0][0]], [a[0][1]], color=c)


def draw_polygon(Polygon, c="red"):#рисование многоугольника (многоугольник - массив точек)
    #draw_dot(Polygon.pol[0].dot)
    for cnt in range(P - 1):
        draw_line_arc(Polygon.pol[cnt + 1].dot, Polygon.pol[cnt].dot, c)
    draw_line_arc(Polygon.pol[P - 1].dot, Polygon.pol[0].dot, c)


def draw_polygons(Polygons, c="red"):
    for Polygon in Polygons:
        draw_polygon(Polygon, c)

# Преобразования
def move_a_dot(a): #поворот точки относительно начальной вершины на 2π/Q (преобразование a)
    center = proj(np.array([[0, RAD]]))
    a_proj = proj(a)
    a_proj = a_proj @ shift_h(center, ZERO_H)
    a_proj = a_proj @ rotate_3d(2 * math.pi / Q)
    a_proj = a_proj @ shift_h(ZERO_H, center)
    ans = proj_r(a_proj)
    return ans


def move_b_dot(a):#поворот точки относительно середины стороны на π (преобразование b)
    MID_proj = proj(MID)
    a_proj = proj(a)
    a_proj = a_proj @ shift_h(MID_proj, ZERO_H)
    a_proj = a_proj @ rotate_3d(math.pi)
    a_proj = a_proj @ shift_h(ZERO_H, MID_proj)
    ans = proj_r(a_proj)
    return ans


def do_word_dot(word, dot):# построение точки по слову
    ans = move_b_dot(move_b_dot(dot))
    for i in range(len(word)):
        m = str(word)
        if m[i] == 'a':
            ans = move_a_dot(ans)
        else:
            ans = move_b_dot(ans)
    return ans


dot0 = np.array([[0, RAD]])

# Это класс "вершина". В нем лежит информация о координатах вершины,
# о слове, по которому она строится из начальной вершины (начальная вершина - центр поворота a),
# и о типе, от которого зависит количество многоугольников во флаге этой вершины:
# тип 0 - вершина не строит флаг
# тип 1 - вершина строит флаг из Q - 2 многоугольников
# тип 2 - вершина строит флаг из Q - 3 многоугольников
# вершина всегда рассматривается не сама по себе, а в контексте многоугольника, которому принадлежит
class vertex:
    def __init__(self, type, word):
        self.type = type
        self.word = word
        self.dot = do_word_dot(word, dot0)

    def print(self):
        print("dot=", self.dot, "_type=", self.type, "_word=", self.word)


ver0 = vertex(1, dot0)


def move_a_ver(a): # поворот вершины относительно начальной вершины на 2π/Q (преобразование a)
    ans = vertex(1, a.word + "a")
    return ans


def move_b_ver(a): # поворот вершины относительно середины стороны на π (преобразование b)
    ans = vertex(1, a.word + "b")
    return ans


def do_word_ver(word, a): # построение вершины по слову
    return vertex(1, a.word + word)

# Это класс "многоугольник", который содержит информацию о наборе вершин многоугольника и
# о слове, по которому он строится из исходного многоугольника
class polygon:
    def __init__(self, word):
        self.word = word
        self.pol = [do_word_ver(word, vertex(1, "ab"*i)) for i in range(P)]


Polygon0 = polygon("")


POLYGONS = [Polygon0]


def pol_out(Polygon): # вывод данных о многоугольнике
    for i in range(len(Polygon.pol)):
        print(i)
        Polygon.pol[i].print()


def move_a_pol(Polygon):#поворот многоугольника относительно начальной вершины на 2π/Q (преобразование a)
    return polygon(Polygon.word + "a")


def move_b_pol(Polygon):#поворот многоугольника относительно середины стороны на π (преобразование b)
    return polygon(Polygon.word + "b")


def do_word_pol(word, Polygon): # построение многоугольника по слову
    return polygon(Polygon.word + word)


def do_flag(Polygon, num): # Эта функция строит флаг - фигуру, получаемую неоднократными поворотами многоугольника на 2π/Q относительно
    Flag_Polygons = [] # вершины, принадлежащей ему. Флаги строятся так, что образуют слои замощения, не перекрываясь
    ver = Polygon.pol[num]
    if ver.type == 0:
        return []
    else:
        for i in range(Q - ver.type - 1):
            Flag_Polygons += [do_word_pol("a"*(i + 1), Polygon0)]
        for i in range(Q - ver.type - 1):
            Flag_Polygons[i] = do_word_pol("ab"*num, Flag_Polygons[i])
        for i in range(Q - ver.type - 1):
            Flag_Polygons[i] = do_word_pol(Polygon.word, Flag_Polygons[i])
        if(len(Flag_Polygons) > 0):
            Flag_Polygons[0].pol[P - 2].type = 0
        for i in range(Q - ver.type - 1):
            Flag_Polygons[i].pol[0].type = 0
            Flag_Polygons[i].pol[P - 1].type = 0
            Flag_Polygons[i].pol[1].type = 2
        if(P == 3):
            Flag_Polygons[0].pol[1].type = 0
            Flag_Polygons[Q - ver.type - 2].pol[1].type = 3
    return Flag_Polygons


def dist(a):
    return math.sqrt(a[0][1]*a[0][1] + a[0][0]*a[0][0])


def dist_layer(LAYER):
    max = -1
    for i in range(len(LAYER)):
        for j in range(P):
            if dist(LAYER[i].pol[j].dot) > max:
                max = dist(LAYER[i].pol[j].dot)
    return max

print("Считаю...")

LAST_LAYER = POLYGONS[:]

LAYERS = [LAST_LAYER]


for i in range(N):
    print("Считаю:    Слой ", i + 1, " из ",  N)
    NEXT_LAYER = []
    NEXT_LAYER
    for Polygon in LAST_LAYER:
        for cnt in range(P):
            if Polygon.pol[cnt].type > 0:
                flag = do_flag(Polygon, cnt) # строит новый слой из флагов вершин старого слоя
                NEXT_LAYER += flag
                if DEBUG_MODE: # рисует и выводит каждый отдельный флаг в режиме отладки
                    if 1: # дополнительное условие вывода флага
                        draw_circle(Polygon.pol[cnt].dot, 0.03)
                        draw_polygons(LAST_LAYER)
                        draw_polygons(NEXT_LAYER)
                        draw_polygon(Polygon, "blue")
                        draw_polygons(flag, "green")
                        plt.show()
                        fig = plt.figure("Polygons", figsize=(10, 10))
                        plt.axes(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1))
    if DEBUG_MODE:
        draw_polygons(NEXT_LAYER)
        plt.show()
        fig = plt.figure("Polygons", figsize=(10, 10))
        plt.axes(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1))
    LAST_LAYER = NEXT_LAYER
    POLYGONS += NEXT_LAYER
    LAYERS += [NEXT_LAYER]
    print("Количество многоугольников в слое:", len(NEXT_LAYER))



draw_circle(np.array([[0, 0]]), 1)

print("Рисую...")
L = sum(len(LAY) for LAY in LAYERS)
cnt = 0
for i in range(len(LAYERS)):
    for j in range(len(LAYERS[len(LAYERS) - 1 - i])):
        cnt += 1
        print("Рисую:   ", int(100*float(cnt)/L), "% [", int(100*float(cnt)/L)*"█", (100 - int(100*float(cnt)/L))*"-", "]")
        draw_polygon(LAYERS[len(LAYERS) - 1 - i][j], colors[(len(LAYERS) - i + 2) % 5])
print("Рисую:   ", "100 % [", 100*"█", "]")
print("Всего многоугольников:", len(POLYGONS))
plt.show()
plt.axis('off')
plt.savefig('i_3_12.svg', transparent=0, bbox_inches='tight', format='svg')

