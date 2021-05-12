# Python Program to find the factors of a number
num = 500
def factors(x):
   print("This program prints the factors of a number \n")
   print("The factors of",x,"are:")
   for i in range(1, x + 1):
       if x % i == 0:
           print(i)


factors(num)
