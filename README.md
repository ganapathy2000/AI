# AI
1.Write a program to implement breadth first search using python.<br>
graph = {<br>
'1' : ['2','10'],<br>
'2' : ['3','8'],<br>
'3' : ['4'],<br>
'4' : ['5','6','7'],<br>
'5' : [],<br>
'6' : [],<br>
'7' : [],<br>
'8' : ['9'],<br>
'9' : [],<br>
'10' : []<br>
}<br>
visited = []<br>
queue = []<br>
def bfs(visited, graph, node):<br>
visited.append(node)<br>
queue.append(node)<br>
while queue:<br>
m = queue.pop(0)<br>
print (m, end = " ")<br>
for neighbour in graph[m]:<br>
if neighbour not in visited:<br>
visited.append(neighbour)<br>
queue.append(neighbour)<br>
print("Following is the Breadth-First Search")<br>
bfs(visited, graph, '1')<br>
Output:-<br>
Following is the Breadth-First Search<br>
1 2 10 3 8 4 9 5 6 7<br>
2.write a program to implement a deapth first search using python.
graph = {<br>
'5' : ['3','7'],<br>
'3' : ['2', '4'],<br>
'7' : ['6'],<br>
'6': [],<br>
'2' : ['1'],<br>
'1':[],<br>
'4' : ['8'],<br>
'8' : []<br>
}<br>
visited = set()<br>
def dfs(visited, graph, node):<br>
if node not in visited:<br>
print (node)<br>
visited.add(node)<br>
for neighbour in graph[node]:<br>
dfs(visited, graph, neighbour)<br>
print("Following is the Depth-First Search")<br>
dfs(visited, graph, '5')<br>
Output:-<br>
Following is the Depth-First Search<br>
5
3
2
1
4
8
7
6
<br>
3.write a program to implement water jug problem using python.<br>
from collections import defaultdict<br>
jug1, jug2, aim = 4, 3, 2<br>
visited = defaultdict(lambda: False)<br>
def waterJugSolver(amt1, amt2):<br>
if (amt1 == aim and amt2 == 0) or (amt2 == aim and amt1 == 0):<br>
print(amt1, amt2)<br>
return True<br>
if visited[(amt1, amt2)] == False:<br>
print(amt1, amt2)<br>
visited[(amt1, amt2)] = True<br>
return (waterJugSolver(0, amt2) or<br>
waterJugSolver(amt1, 0) or<br>
waterJugSolver(jug1, amt2) or<br>
waterJugSolver(amt1, jug2) or<br>
waterJugSolver(amt1 + min(amt2, (jug1-amt1)),<br>
amt2 - min(amt2, (jug1-amt1))) or<br>
waterJugSolver(amt1 - min(amt1, (jug2-amt2)),<br>
amt2 + min(amt1, (jug2-amt2))))<br>
else:<br>
return False<br>
print("Steps: ")<br>
waterJugSolver(0, 0)<br>

Output:-<br>
Steps:
0 0
4 0
4 3
0 3
3 0
3 3
4 2
0 2
True<br>

4.write a program to implement tower of hannoi using python.<br>
def TowerOfHanoi(n , source, destination, auxiliary):<br>
if n==1:<br>
print ("Move disk 1 from source",source,"to destination",destination)<br>
return<br>
TowerOfHanoi(n-1, source, auxiliary, destination)<br>
print ("Move disk",n,"from source",source,"to destination",destination)<br>
TowerOfHanoi(n-1, auxiliary, destination, source)<br>
n = 4<br>
TowerOfHanoi(n,'A','B','C')<br>

Output:-
Move disk 1 from source A to destination C
Move disk 2 from source A to destination B
Move disk 1 from source C to destination B
Move disk 3 from source A to destination C
Move disk 1 from source B to destination A
Move disk 2 from source B to destination C
Move disk 1 from source A to destination C
Move disk 4 from source A to destination B
Move disk 1 from source C to destination B
Move disk 2 from source C to destination A
Move disk 1 from source B to destination A
Move disk 3 from source C to destination B
Move disk 1 from source A to destination C
Move disk 2 from source A to destination B
Move disk 1 from source C to destination B

5.write a program to implement best first search using python.
from queue import PriorityQueue
import matplotlib.pyplot as plt
import networkx as nx
