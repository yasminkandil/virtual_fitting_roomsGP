weight=70
height=180
gender='male'
output_coefficient=6
if gender=='male':
    perfect_weight=height-110
else:
    perfect_weight=height-100
weight_diff=weight-perfect_weight
output_weight=-((weight_diff*output_coefficient)/perfect_weight)
print(output_weight)