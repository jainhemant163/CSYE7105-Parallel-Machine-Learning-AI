# Program to check Armstrong numbers in a certain interval

lower = 100
upper = 2000
print("Program to print armstrong numbers in an interval")
for num in range(lower, upper + 1):

   order = len(str(num))
    
   sum = 0

   temp = num
   while temp > 0:
       digit = temp % 10
       sum += digit ** order
       temp //= 10

   if num == sum:
       print(num)
