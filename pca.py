""" PCA
  Beatriz de Camargo Castex Ferreira - 10728077 - USP São Carlos - IFSC
  05/2020

  Nesse programa vamos gerar um padrão circular de pontos uniformemente
  distribuídos e aplicar o PCA

  Referências:
  https://www.researchgate.net/publication/340114268
  https://blogs.sas.com/content/iml/2016/03/30/generate-uniform-2d-ball.html
  https://datascienceplus.com/understanding-the-covariance-matrix/
  http://www.cs.otago.ac.nz/cosc453/student_tutorials/principal_components.pdf

  """

import numpy as np
import matplotlib.pyplot as plt


# 1 - Gerar distribuição circular de pontos uniformemente distribuídos:

# Quantos pontos?
N = 500

# Podemos escolher pontos em cordenadas polares, escolhendo raios e ângulos
# dentro de r = 1 e theta = 2pi.
r = np.sqrt(np.random.uniform(0, 1, N))  # Temos que escolher com uma
# distribuição proporcional à raiz porque queremos uma distribuição 2D, ou seja,
# tem que ser proporcional à pi*rˆ2
theta = np.pi * np.random.uniform(0, 2, N)

# Fazemos a transformação para coordenadas cartesianas.
x = r * np.cos(theta)
y = r * np.sin(theta)

# 2 - Comprimir os dados verticalmente para 1/5:
y = y * 0.2

# Rotacionar pontos usando [cos(30) sin(30); sin(30) cos(30)]:
rotation = [[np.cos(0.523599), np.sin(0.523599)],
            [np.sin(0.523599), np.cos(0.523599)]]
[x, y] = np.dot(rotation, [x, y])
F = np.array([x, y]).T

# 4 - Visualizar os dados, para ver se está parecido com Figura 9, CDT-24
plt.scatter(x, y, color="orange", s=5)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Antes do PCA')
plt.show()

# 5 - Obter matriz de covariância K dos dados gerados:
# Definir função de covariancia


def covariance(X, Y):

    # Obter média de x e y:
    mean_x = np.mean(X)
    mean_y = np.mean(Y)

    # Calcular a covariância:
    return np.sum((X - mean_x) * (Y - mean_y)) / (len(X) - 1)


# Montar a matriz de covariância
K = ([covariance(x, x), covariance(x, y)],
     [covariance(y, x), covariance(y, y)])

# 6 - Obter autovalores/autovetores de K
eig_values, eig_vectors = np.linalg.eig(K)

# 7 - Ordenar decrescentemente os autovalores juntamente com autovalores
eig_values_dec = np.sort(eig_values)[::-1]

# 8 - Obter matriz Q usando autovetores como linhas

Q = eig_vectors

for o in range(len(eig_values)):
    for u in range(len(eig_values_dec)):
        if (eig_values_dec[u] == eig_values[o]):
            Q[u] = eig_vectors[o]


# 9 - Aplicar nos dados e mostrar novo resultado.
[x, y] = np.dot(Q.T, F.T)

plt.scatter(x, y, color="orange", s=5)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Depois do PCA')
plt.show()
