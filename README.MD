# Report

## Abstract
Here is a sample of technical writing from a research problem report. The behavior of a mechanical system was considered, which consisted of two pistons linked by springs with different stiffness. The movement of the pistons was constrained in the vertical direction. As a result, the relationship between the displacement of the second piston and the first piston was described.

## Initiation of the problem



## Force Displacement curve (FD-curve)
We can obtain the force at piston due the spring deformation using the following equations:

Spring length at initial moment:     
$$ L = \sqrt{L_x^2 + L_y^2} $$

Length after deformation:    
$$ L_d = \sqrt{L_x^2 + (L_y - d)^2} $$

$d$ -- displacement from a position of relaxation

The force induced by spring deformation is a linear function:
$$ F=kx$$
where 

$k$ -- spring stiffness

$x$ -- spring deformation

Therefore, we can get the spring force:    
$$F_s = k(L - L_d)$$

Finally, force at the piston is evaluated by:     
$$F_d = F_s(L_y - d)/L_d$$

For the sake of clarity, the ratio $(L_y - d)/L_d$ can be expressed as $sin(\alpha)$, where $\alpha$ is an angle between the spring and the horizontal line.

The last equation represent the Force Displacement curve (FD-curve) of the piston.