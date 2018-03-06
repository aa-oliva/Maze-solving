# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 23:14:22 2018

@author: aa-ol  
"""


from tkinter import filedialog
from tkinter import *
from PIL import Image
import numpy as np
from scipy import stats
import heapq
import random
import collections

#Iniciar TKinter image
root = Tk()

#devuelve la ruta del archivo a seleccionar
def abrir_imagen():
    root.filename = filedialog.askopenfilename(initialdir="/", title="Select file",
                                               filetypes=(("all files", "*.*"), ("png files","*.png")))     
    return root.filename


#devuelve la ruta del archivo a guardar
def guardar_imagegn():
    root.filename =  filedialog.asksaveasfilename(initialdir = "/",title = "Select file",
                                                  filetypes = (("all files","*.*"),("png files","*.png")))
    return root.filename
    


###############################################################################
###############################################################################
#Se crean los draw 
def from_id_width(id, width):
    return (id % width, id // width)


def draw_tile(graph, id, style, width):
    r = "."
    if 'number' in style and id in style['number']: r = "%d" % style['number'][id]
    if 'point_to' in style and style['point_to'].get(id, None) is not None:
        (x1, y1) = id
        (x2, y2) = style['point_to'][id]
        if x2 == x1 + 1: r = "\u2192"
        if x2 == x1 - 1: r = "\u2190"
        if y2 == y1 + 1: r = "\u2193"
        if y2 == y1 - 1: r = "\u2191"
        #Se Puede utilizar diagonales
        '''
        if y2 == y1 - 1 and x2 == +1: r = "\u2196"
        if y2 == y1 - 1 and x2 == -1: r = "\u2197"
        if y2 == y1 + 1 and x2 == -1: r = "\u2198"
        if y2 == y1 + 1 and x2 == +1: r = "\u2199"
        '''
        
    if 'start' in style and id == style['start']: r = "A"
    if 'goal' in style and id == style['goal']: r = "Z"
    if 'path' in style and id in style['path']: 
        r = "@"
        Matrix[id]=4

    if id in graph.walls: r = "#" * width
    return r


def draw_grid(graph, width=2, **style):
    for y in range(graph.height):
        for x in range(graph.width):
            print("%%-%ds" % width % draw_tile(graph, (x, y), style, width), end="")
            
        #print()



class SquareGrid:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.walls = []

    def in_bounds(self, id):
        (x, y) = id
        return 0 <= x < self.width and 0 <= y < self.height

    def passable(self, id):
        return id not in self.walls

    def neighbors(self, id):
        (x, y) = id
        results = [(x + 1, y), (x, y - 1), (x - 1, y), (x, y + 1)] #,(x+1,y+1), (x+1,y-1), (x-1,y+1), (x-1,y-1)]
        if (x + y) % 2 == 0: results.reverse()  # aesthetics
        results = filter(self.in_bounds, results)
        results = filter(self.passable, results)
        return results


class GridWithWeights(SquareGrid):
    def __init__(self, width, height):
        super().__init__(width, height)
        self.weights = {}

    def cost(self, from_node, to_node):
        return self.weights.get(to_node, 10)


class Queue:
    def __init__(self):
        self.elements = collections.deque()
    
    def empty(self):
        return len(self.elements) == 0
    
    def put(self, x):
        self.elements.append(x)
    
    def get(self):
        return self.elements.popleft()

class PriorityQueue:
    def __init__(self):
        self.elements = []

    def empty(self):
        return len(self.elements) == 0

    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        return heapq.heappop(self.elements)[1]


def reconstruct_path(came_from, start, goal):
    current = goal
    path = [current]
    while current != start:
        current = came_from[current]
        path.append(current)
    path.append(start)  # optional
    path.reverse()  # optional
    return path

###############################################################################
###############################################################################

#Euristicas
def heuristic(a, b):
    (x1, y1) = a
    (x2, y2) = b
#Manhattan
    return abs(x1 - x2) + abs(y1 - y2)

def heuristic2(a, b):
    (x1, y1) = a
    (x2, y2) = b
#Hamming
    return abs(x1 - x2)**abs(y1 - y2)


###############################################################################
#Algoritmos de busqueda
def Aestrella(graph, start, goal):
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0

    while not frontier.empty():
        current = frontier.get()

        if current == goal:
            break

        for next in graph.neighbors(current):
            new_cost = cost_so_far[current] + graph.cost(current, next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristic(goal, next)
                frontier.put(next, priority)
                came_from[next] = current

    return came_from, cost_so_far

def Aestrella_v2(graph, start, goal):
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0

    while not frontier.empty():
        current = frontier.get()

        if current == goal:
            break

        for next in graph.neighbors(current):
            new_cost = cost_so_far[current] + graph.cost(current, next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristic2(goal, next)
                frontier.put(next, priority)
                came_from[next] = current

    return came_from, cost_so_far

def bfs(graph, start, goal):
    frontier = Queue()
    frontier.put(start)
    came_from = {}
    came_from[start] = None
    
    while not frontier.empty():
        current = frontier.get()
        
        if current == goal:
            break
        
        for next in graph.neighbors(current):
            if next not in came_from:
                frontier.put(next)
                came_from[next] = current
    
    return came_from















"""
Se crea el mapa
"""

#leer la imagen
im = Image.open(abrir_imagen())

#col = ancho, row = alto
col,row =  im.size
print (col,row)

#Tamaño del grid
n=20


#Se calcula la cantidad de pixeles por cuadro, para la cuadricula ingresada
height = int( row/n)
weight = int(col/n)

#Se inicializan las posiciones de inicio
iniciox=0
inicioy=0

#Se inicializan las posiciones de meta
metax=0
metay=0


#Se crea el mapa
gs = SquareGrid(n, n)
mapa = GridWithWeights(n, n)
#Se le da el peso a las casillas
mapa.weights = {loc: random.randint(1,1) for loc in [()]}


Matrix =np.ndarray((n, n))

r= []
g= []
b= []
#color de pixel en una matriz
pixels = im.load()


'''Comienza aqui prueba con la moda'''
cont3=0
for nn in range (n):
    cont2=0
    for ii in range (n):
        
        for i in range (height):
            cont = 0
            #for para ingresar rgb a cada lista y sacarle la moda por casilla
            for j in range (weight):
                rl, gl, bl = pixels[(j+cont+cont3),(i+cont2)]
                r.append(rl), g.append(gl), b.append(bl) 
                
            #se agrega el salto 
            cont+=weight
            
        rl = stats.mode(r)
        gl = stats.mode(g)
        bl = stats.mode(b)
        
        #Se crea la matriz para saber los colores del grid nXn
        #camino o blanco
        if (rl[0]==255 and gl[0]==255 and bl[0]==255):
            Matrix[ii,nn]=1
        #rojo inicio
        elif (rl[0]>=229 and gl[0]<=50 and bl[0]<=50):
            Matrix[ii,nn]=2
        #verde llegada
        elif (rl[0]<=181 and gl[0]>=229 and bl[0]<=50):
            Matrix[ii,nn]=3
        #pared o negro
        elif (rl[0]==0 and gl[0]==0 and bl[0]==0):
            Matrix[ii,nn]=0
           
        
          
        #Se utiliza la matriz de colores del grid nXn para hacer la discretizacion con el camino.
        for i in range (height):
            cont=0
            #for para llenar la casilla con el rgb de la moda
            for j in range (weight):
                #camino
                if (Matrix[ii,nn]==1):
                    pixels[(j+cont+cont3),((i+cont2))] = 255 ,255 ,255
                    
                #inicio
                elif (Matrix[ii,nn]==2):
                    pixels[(j+cont+cont3),((i+cont2))] = 255 ,0 ,0
                    iniciox=nn
                    inicioy=ii
                    
                #llegada
                elif (Matrix[ii,nn]==3):
                    pixels[(j+cont+cont3),((i+cont2))] = 0 ,255 ,0
                    metax=nn
                    metay=ii
                    
                #path
                elif (Matrix[ii,nn]==4):
                    pixels[(j+cont+cont3),((i+cont2))] = 0 ,0 ,255
                           
                #pared
                elif(Matrix[ii,nn]==0):
                    pixels[(j+cont+cont3),((i+cont2))] = 0 ,0 ,0
                    #se agregan las paredes
                    mapa.walls += [(ii,nn)]
                    gs.walls +=[(ii,nn)]
        
            #se agrega el salto 
            cont+=weight
            
            
            
       # se reinician las listas rgb por casilla para tener una moda exacta
        r= []
        g= []
        b= []
        
        #Se hace el salto a la derecha
        cont2+=height  
    #se hace el salto para abajo
    cont3+=weight

#transformar de data a imagen
data = np.zeros((col, row, 3), dtype=np.uint8)

#matriz de pixeles (color)
for i in range(row):
    for j in range(col):
        r,g,b =  pixels[j,i]
        data[j,i]= r,g,b
    
#transformar de data a imagen
img = Image.fromarray(data, 'RGB')
#Guardar imagen 
img.save('discretizacion.png')

print ("Se prueba A* Manhattan")
print ("mapa de hijos")
came_from, cost_so_far = Aestrella(mapa, (inicioy, iniciox), (metay, metax))
draw_grid(mapa, width=3, point_to=came_from, start=(inicioy, iniciox), goal=(metay, metax))
print ()
print("pesos")
draw_grid(mapa, width=3, number=cost_so_far, start=(inicioy, iniciox), goal=(metay, metax))
print ()
print("Solucion")
draw_grid(mapa, width=3, path=reconstruct_path(came_from, start=(inicioy, iniciox), goal=(metay, metax)))


#se agrega el color path a la imagen final
cont3=0
for nn in range (n):
    cont2=0
    for ii in range (n):
        #Se utiliza la matriz de colores del grid nXn para hacer la discretizacion con el camino.
        for i in range (height):
            cont=0
            #for para llenar la casilla con el rgb de la moda
            for j in range (weight):
                if (Matrix[ii,nn]==4):
                    pixels[(j+cont+cont3),((i+cont2))] = 0 ,0 ,255
                    
        
            #se agrega el salto 
            cont+=weight
        #Se hace el salto a la derecha
        cont2+=height  
    #se hace el salto para abajo
    cont3+=weight
###############################################################################

#transformar de data a imagen
data = np.zeros((col, row, 3), dtype=np.uint8)

#matriz de pixeles (color)
for i in range(row):
    for j in range(col):
        r,g,b =  pixels[j,i]
        data[j,i]= r,g,b
        
        
#transformar de data a imagen
img = Image.fromarray(data, 'RGB')
#Guardar imagen
img.save('AEstrella-Manhattan_0.png')
#limpieza
for i in range(row):
    for j in range(col):
        
        r,g,b =  pixels[j,i]
        if (r==0 and g==0 and b==255):
            pixels[j,i]= 0,0,255
            
        r,g,b =  pixels[j,i]
        data[j,i]= r,g,b

###############################################################################

"""
Se prueba con Aestrella version2 
"""
print ("Se prueba Aestrella version2 ")
print ()
print ("mapa de hijos")
came_from, cost_so_far = Aestrella_v2(mapa, (inicioy, iniciox), (metay, metax))
draw_grid(mapa, width=3, point_to=came_from, start=(inicioy, iniciox), goal=(metay, metax))
print ()
print("pesos")
draw_grid(mapa, width=3, number=cost_so_far, start=(inicioy, iniciox), goal=(metay, metax))
print ()
print("solucion")
draw_grid(mapa, width=3, path=reconstruct_path(came_from, start=(inicioy, iniciox), goal=(metay, metax)))

#se agrega el color path a la imagen final
cont3=0
for nn in range (n):
    cont2=0
    for ii in range (n):
        #Se utiliza la matriz de colores del grid nXn para hacer la discretizacion con el camino.
        for i in range (height):
            cont=0
            #for para llenar la casilla con el rgb de la moda
            for j in range (weight):
                if (Matrix[ii,nn]==4):
                    pixels[(j+cont+cont3),((i+cont2))] = 0 ,0 ,255
            #se agrega el salto 
            cont+=weight
        #Se hace el salto a la derecha
        cont2+=height  
    #se hace el salto para abajo
    cont3+=weight
###############################################################################

#transformar de data a imagen
data = np.zeros((col, row, 3), dtype=np.uint8)

#matriz de pixeles (color)
for i in range(row):
    for j in range(col):
        r,g,b =  pixels[j,i]
        data[j,i]= r,g,b
        
        
#transformar de data a imagen
img = Image.fromarray(data, 'RGB')
#Guardar imagen
img.save('AEstrella-V2_0.png')


###############################################################################

print ("Se prueba BFS")
parents = bfs(gs, (inicioy, iniciox), (metay, metax))
draw_grid(gs, width=3, point_to=parents, start=(inicioy, iniciox), goal=(metay, metax))
draw_grid(gs, width=3, path=reconstruct_path(came_from, start=(inicioy, iniciox), goal=(metay, metax)))


#se agrega el color path a la imagen final
cont3=0
for nn in range (n):
    cont2=0
    for ii in range (n):
        #Se utiliza la matriz de colores del grid nXn para hacer la discretizacion con el camino.
        for i in range (height):
            cont=0
            #for para llenar la casilla con el rgb de la moda
            for j in range (weight):
                if (Matrix[ii,nn]==4):
                    pixels[(j+cont+cont3),((i+cont2))] = 0 ,0 ,255    
            #se agrega el salto 
            cont+=weight
        #Se hace el salto a la derecha
        cont2+=height  
    #se hace el salto para abajo
    cont3+=weight
###############################################################################

#transformar de data a imagen
data = np.zeros((col, row, 3), dtype=np.uint8)

#matriz de pixeles (color)
for i in range(row):
    for j in range(col):
        r,g,b =  pixels[j,i]
        data[j,i]= r,g,b
        
        
#transformar de data a imagen
img = Image.fromarray(data, 'RGB')
#Guardar imagen
img.save('BFS_0.png')


###############################################################################
quit()