# Report

https://github.com/octopus-vulgaris/research_problem

_play `research_problem.ipynb`_

## Abstract
Here is a __sample of technical writing__ from a research problem report. The behavior of a mechanical system was considered, which consisted of two pistons linked by springs with different stiffness. The movement of the pistons was constrained in the vertical direction. As a result, the relationship between the displacement of the second piston and the first piston was described.

## Initiation of the problem



## Force Displacement curve (FD-curve)
We can obtain the force at piston due the spring deformation using the following equations:

Spring length at initial moment:     
$$L = \sqrt{L_x^2 + L_y^2}$$

Length after deformation:    
$$L_d = \sqrt{L_x^2 + (L_y - d)^2}$$

$d$ -- displacement from a position of relaxation

The force induced by spring deformation is a linear function:
$$F=kx$$
where 

$k$ -- spring stiffness

$x$ -- spring deformation

Therefore, we can get the spring force:    
$$F_s = k(L - L_d)$$

Finally, force at the piston is evaluated by:     
$$F_d = F_s(L_y - d)/L_d$$

For the sake of clarity, the ratio $(L_y - d)/L_d$ can be expressed as $sin(\alpha)$, where $\alpha$ is an angle between the spring and the horizontal line.

The last equation represent the Force Displacement curve (FD-curve) of the piston.

![FD-curve](/img/Figure_1.png)

## Two piston system behave

To understand the kinematics of the system, we will use the principle of equilibrium of Y-forces. We will vary the position of the second piston and, for each position, find the coordinate of the first piston such that the Y-forces induced by the deformed springs are in equilibrium.

Follow the approach above describe the behave of the system. Observe the relaxation position.

![figure](/img/fig_2.png)

Slider sets the coordinate of the second piston (`var`), `root` variable shows the coordinate of the first one calculated as roots of the $F_{d_1} - F_{d_2} = 0$ equation. Using the $F(d)$ function we can visually determine the force acting on the first piston at the `root` position.

![figure](/img/fig_3.png)

At the position shown on picture above we can see that the system have three points of equal Y-forces. Two of them are stable and one at $"0"$ coordinate unstable. The domain of the root in the positive half of the plane is unreachable if there are no dynamical oscillation processes of the free piston. However, the piston can reach this domain if it moves from down to up.

![figure](/img/fig_4.png)

When the curve of the 1st piston position deforms to a single-root form, the formation of a new root occurs on the opposite branch, and the solution jumps to the right father point. The behavior of the system is specific to this threshold. The switch triggers a change in the force direction on the second piston, which in turn causes the springs to move the system towards the relaxed down position. Similar behavior can be observed when the system reaches the first potential threshold, but without the first piston jumping. Another specific feature of the system is that the potential barrier has a maximum derivative before reaching its own maximum when the system starts to move towards the next stable state.

The system has four stable states, characterized by equal spring lengths. The coordinates of these states are
$$(y_1, y_2) \in \{ (1, 2), (-1, 0), (-1, -2), (1, 0) \}$$

The graph of states is shown below.

![graph](/img/graph.png)


## Conclusion

The system with a free massless piston and two springs have described. The specific of the kinematics revealed as switching behave has obtained.

## Application

Hmmm.... At this point, I cannot exactly imagine how it may be used to create passive functionality. However, if extra conditions are known, this nonlinearity, with some modification, can be used for specific tasks. At the very least, we need to know which parameter needs control, whether it be the position of pistons, force acting on pistons, configuration of the system, switches between stable states, etc.
Undoubtedly, it may be used for modeling objects that behave similarly, as well as for non-linear suspension systems and for dampening shocks and vibrations.