# > Proyecto 4
# Cesar Quesada Zamora 
# B76041

# Bibliotecas de interés para la simulación 

from PIL import Image
import numpy as np
import time
import matplotlib.pyplot as plt

#....... PARTE A ........

# 1. Función que transforma la imagen al modelo de colores RGB

def fuente_info(imagen):
   
    img = Image.open(imagen)
    
    return np.array(img)

# 2.Función que transforma el modelo de colores RGB en una cadena de bits

def rgb_a_bit(imagen):
   
    # Obtener las dimensiones de la imagen
    x, y, z = imagen.shape
    
    # Número total de pixeles
    n_pixeles = x * y * z

    # Convertir la imagen a un vector unidimensional de n_pixeles
    pixeles = np.reshape(imagen, n_pixeles)

    # Convertir los canales a base 2
    bits = [format(pixel,'08b') for pixel in pixeles]
    bits_Rx = np.array(list(''.join(bits)))
    
    return bits_Rx.astype(int)


# 3.Función que crea la onda modulada en QPSK 

def modulador(bits, fc, mpp):
   
    #  Parámetros de la 'señal' de información (bits)
    
    N = len(bits) # Cantidad de bits

    # Construyendo un periodo de la señal portadora s(t)
    
    Tc = 1 / fc  # periodo [s]
    t_periodo = np.linspace(0, Tc, mpp)
    portadora_1 = np.cos(2*np.pi*fc*t_periodo)  # Portadora I
    portadora_2 = np.sin(2*np.pi*fc*t_periodo)  # Portadora Q

    # Inicializar la señal modulada s(t)
    
    t_simulacion = np.linspace(0, N*Tc, N*mpp) # Cada unidad de tiempo representa un periodo de muestreo
    senal_I = np.zeros(t_simulacion.shape)     # Vector vacío de la señal I
    senal_Q = np.zeros(t_simulacion.shape)     # Vector vacío de la señal Q
    moduladoraI = np.zeros(t_simulacion.shape)  # señal de información a partir de I 
    moduladoraQ = np.zeros(t_simulacion.shape)  # señal de información a partir de Q
   
    # Asignar las formas de onda según los bits (QPSK)
    j = 0    # Contador de muestreos 
    
    # La idea es recorrer el vector de bits de dos en dos para asignar el valor de A1 y A2
    # Bit b1 --> A1 | Bit b2 ---> A2  
    # Se obtienen una moduladora por cada portadora 
   
    for i in range(0,N,2):
        
        # Portadora I
    
        if bits[i] == 1:
            senal_I[j*mpp : (j+1)*mpp] = portadora_1
            moduladoraI[j*mpp : (j+1)*mpp] = 1
           
        else:
            senal_I[j*mpp : (j+1)*mpp] = portadora_1 * -1
            moduladoraI[j*mpp : (j+1)*mpp] = 0
            
        # Portadora Q
        
        if i < N:  # Este factor evita que exista un sobrepaso en la escritura del array 
            
            if bits[i+1] == 1:                              
                senal_Q[j*mpp : (j+1)*mpp] = portadora_2
                moduladoraQ[j*mpp : (j+1)*mpp] = 1
               
            else:
                senal_Q[j*mpp : (j+1)*mpp] = portadora_2 * -1
                moduladoraQ[j*mpp : (j+1)*mpp] = 0
        j= j+1
        
              
    senal_Tx =  senal_I + senal_Q  # Portadora Final S(t) 
    
    
    #  Calcular la potencia promedio de la señal modulada
    
    Pm = (1 / (N*Tc)) * np.trapz(pow(senal_Tx, 2), t_simulacion)
    
    return senal_Tx, Pm, portadora_1,portadora_2 , moduladoraI, moduladoraQ, t_simulacion


# 4.Función que simula un medio no ideal (ruidoso)

def canal_ruidoso(senal_Tx, Pm, SNR):
  
    # Potencia del ruido generado por el canal
    Pn = Pm / pow(10, SNR/10)

    # Generando ruido auditivo blanco gaussiano
    ruido = np.random.normal(0, np.sqrt(Pn), senal_Tx.shape)

    # Señal distorsionada por el canal ruidoso
    senal_Rx = senal_Tx + ruido

    return senal_Rx

# 5.Función que demodula la señal trasmitida por el medio ruidoso 

def demodulador(senal_Rx, portadoraI,portadoraQ, mpp):
 
    # Cantidad de muestras en senal_Rx
    M = len(senal_Rx)

    # Cantidad de bits en transmisión
    N =  int(M / mpp) # Se multiplica por dos ya que hay dos bits por muestreo

    # Vector para bits obtenidos por la demodulación
    bits_Rx = np.zeros(N)

    # Vector para la señal demodulada
    senal_demoduladaI = np.zeros(M)
    senal_demoduladaQ = np.zeros(M)
   
    # Demodulación
    # Se ejecuta una demodulación por cada señal portadora 
    
    j =0 # Puntero del array de bits
    
    for i in range(N):
        
            # Producto interno de dos funciones
            producto_1 = senal_Rx[(i)*mpp : (i+1)*mpp] * portadoraI                   # Producto de la portadora I
            producto_2 = senal_Rx[(i)*mpp : (i+1)*mpp] * portadoraQ                   # Producto de la portadora Q
            senal_demoduladaI[i*mpp : (i+1)*mpp] = producto_1        # Parte asociada a la portadora I
            senal_demoduladaQ[(i)*mpp : (i+1)*mpp] = producto_2      # Parte asociada a la portadora Q
            Ep_1 = np.sum(producto_1) 
            Ep_2 = np.sum(producto_2) 
    
            # Criterio de decisión por detección de energía
            
            # Bit asociado a la portadora I
            if j < N:
                if Ep_1 > 0:
                    bits_Rx[j] = 1
                else:
                    bits_Rx[j] = 0
                
            # Bit asociado a la portadora Q
           
                if Ep_2 > 0:
                    bits_Rx[j+1] = 1
                else:
                    bits_Rx[j+1] = 0

            j = j+2 # Se recorre de dos en dos
            
    return bits_Rx.astype(int), senal_demoduladaI,  senal_demoduladaQ

# 6.Función que reconstruye la imagen 

def bits_a_rgb(bits_Rx, dimensiones):

    # Cantidad de bits
    N = len(bits_Rx)

    # Se reconstruyen los canales RGB
    bits = np.split(bits_Rx, N / 8)

    # Se decofican los canales:
    canales = [int(''.join(map(str, canal)), 2) for canal in bits]
    pixeles = np.reshape(canales, dimensiones)

    return pixeles.astype(np.uint8)


# 7.Figuras de la modulación

# Parámetros
fc = 5000  # frecuencia de la portadora
mpp = 20   # muestras por periodo de la portadora
SNR = 5    # relación señal-a-ruido del canal

# Iniciar medición del tiempo de simulación
inicio = time.time()

# 1. Importar y convertir la imagen a trasmitir
imagen_Tx = fuente_info('arenal.jpg')
dimensiones = imagen_Tx.shape

# 2. Codificar los pixeles de la imagen
bits_Tx = rgb_a_bit(imagen_Tx)

# 3. Modular la cadena de bits usando el esquema BPSK
senal_Tx, Pm, portadoraI, portadoraQ, moduladoraI,moduladoraQ, t_simulacion = modulador(bits_Tx, fc, mpp)

# 4. Se transmite la señal modulada, por un canal ruidoso
senal_Rx = canal_ruidoso(senal_Tx, Pm, SNR)

# 5. Se desmodula la señal recibida del canal
bits_Rx,senal_demoduladaI, senal_demoduladaQ = demodulador(senal_Rx, portadoraI, portadoraQ, mpp)

# Se plotean las distintas etapas del proceso

fig, (ax1,ax2, ax3) = plt.subplots(nrows=3, sharex=True, figsize=(14, 7))

# La onda cuadrada moduladora (bits de entrada)
ax1.plot(moduladoraI[0:600], color='r', lw=2) 
ax1.set_ylabel('$b1(t)$')

# La señal modulada por BPSK
ax2.plot(moduladoraQ[0:600], color='b', lw=2) 
ax2.set_ylabel('$b2(t)$')

# La señal modulada al dejar el canal
ax3.plot(senal_Tx[0:600], color='g', lw=2) 
ax3.set_ylabel('$s(t)$')
ax3.set_xlabel('$t$ / milisegundos')

fig, (ax1,ax2,ax3,ax4) = plt.subplots(nrows=4, sharex=True, figsize=(14, 7))

# La onda cuadrada moduladora (bits de entrada)
ax1.plot(senal_Tx[0:600], color='g', lw=2) 
ax1.set_ylabel('$b(t)$')

# La señal modulada por BPSK
ax2.plot(senal_Rx[0:600], color='m', lw=2) 
ax2.set_ylabel('$s(t) + n(t)$')

# La señal demodulada I
ax3.plot(senal_demoduladaI[0:600], color='r', lw=2) 
ax3.set_ylabel('$b1^{\prime}(t)$')

# La señal demodulada Q
ax4.plot(senal_demoduladaQ[0:600], color='b', lw=2) 
ax4.set_ylabel('$b2^{\prime}(t)$')
ax4.set_xlabel('$t$ / milisegundos')

# 6. Se visualiza la imagen recibida 
imagen_Rx = bits_a_rgb(bits_Rx, dimensiones)
Fig = plt.figure(figsize=(10,6))

# Cálculo del tiempo de simulación
print('Duración de la simulación: ', time.time() - inicio)

# 7. Calcular número de errores
errores = sum(abs(bits_Tx - bits_Rx))
BER = errores/len(bits_Tx)
print('{} errores, para un BER de {:0.4f}.'.format(errores, BER))

# Mostrar imagen transmitida
ax = Fig.add_subplot(1, 2, 1)
imgplot = plt.imshow(imagen_Tx)
ax.set_title('Transmitido')

# Mostrar imagen recuperada
ax = Fig.add_subplot(1, 2, 2)
imgplot = plt.imshow(imagen_Rx)
ax.set_title('Recuperado')
Fig.tight_layout()

plt.imshow(imagen_Rx)


#......... PARTE 2..............

# 1.Tiempo de muestreo 
t_muestra = np.linspace(0, 0.1,100)

# 2.Posibles valores de A
A=[1,-1]

# 3.Formas de onda posibles 
X_t = np.empty((4, len(t_muestra)))	   # 4 funciones del tiempo x(t) 

# 4.Nueva figura 
plt.figure()

# 5. Matriz con los valores de cada función posibles   
for i in A:
    x1 = i * np.cos(2*(np.pi)*fc*t_muestra) +  i* np.sin(2*(np.pi)*fc*t_muestra)
    x2 = -i * np.cos(2*(np.pi)*fc*t_muestra) +  i* np.sin(2*(np.pi)*fc*t_muestra) 
    X_t[i,:] = x1
    X_t[i+1,:] = x2
    plt.plot(t_muestra,x1, lw=2)
    plt.plot(t_muestra, x2, lw=2)       

# 6. Promedio de las 4 realizaciones en cada instante 
P = [np.mean(X_t[:,i]) for i in range(len(t_muestra))]
plt.plot(t_muestra, P, lw=6,color='k',label='Promedio Realizaciones')

# 7. Graficar el resultado teórico del valor esperado
E = np.mean(senal_Tx)*t_muestra  # Valor esperado de la señal 
plt.plot(t_muestra, E, '-.', lw=3,color='c',label='Valor teórico')

# 8. Mostrar las realizaciones, y su promedio calculado y teórico
plt.title('Realizaciones del proceso aleatorio $X(t)$')
plt.xlabel('$t$')
plt.ylabel('$x_i(t)$')
plt.legend()
plt.show()
