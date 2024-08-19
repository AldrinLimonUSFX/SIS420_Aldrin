import random
import csv

def generar_datos(n=100):
    datos = []
    for _ in range(n):
        estatura = round(random.uniform(1.4, 2.05), 2)  
        peso = round(random.uniform(50, 100), 2)  
       
        if estatura < 1.5:
            peso = round(random.uniform(50, 70), 2)
        elif estatura > 1.9:
            peso = round(random.uniform(80, 100), 2)
        datos.append((peso, estatura))
    return datos

datos = generar_datos()

with open('datos_peso_estatura.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Peso', 'Estatura'])  
    writer.writerows(datos) 
    
print("Datos generados y guardados en 'datos_peso_estatura.csv'.")
