# Control Module Overview 

This document describes the aerodynamic control layer in the IronMan CFD backend. It covers:

1. **Purpose & Scope**
2. **Control Surfaces & Actuation**
3. **Control Loop Architecture**
4. **Integration with Solver**
5. **Configuration Parameters**
6. **Usage Examples**

---

## 1. Purpose & Scope

In addition to predicting forces and turbulence (handled by `turbulence_mdoels`), the IronMan suit must actively control:

- **Orientation**: Adjust reaction jets and control surfaces to maintain attitude.
- **Stability**: Damp out oscillations due to gusts or maneuvering.
- **Thrust Vectoring**: Steer propulsion jets for translation and rotation.

This "control" layer computes target surface deflections or jet-flow sepoints, then feeds them back into the flow solver for closed-loop simulation. 

---

## 2. Control Surfaces & Actuation

### 2.1 Control Surfaces 

| Surface Name          | Position            | Degrees of Freedom | Typical Range |
| --------------------- | ------------------- | ------------------ | --------------|
| Flap (left/right)     | Wing trailing edge  | 1 (deflection)     | ±30°          |
| Aileron (left/right)  | Mid-wing span       | 1 (deflection)     | ±20°          |
| Elevator (fore & aft) | Tail section        | 1 (deflection)     | ±25°          |
| Rudder                | Vertical fin        | 1 (deflection)     | ±30°          |

### 2.2 Actuator Model

- **First-order lag**:  
  \[
    \tau_a \frac{d\delta}{dt} + \delta = \delta_\mathrm{cmd}
  \]
  where \(\delta\) is surface deflection, \(\delta_\mathrm{cmd}\) is command, and \(\tau_a\) is actuator time constant.

- **Limits & Saturation**:  
  - Max rate: \(\dot\delta_\max\)  
  - Hard stops at \(\pm \delta_\max\)

---

## 3. Control Loop Architecture

```mermaid
flowchart TD
    UserCmd[User Command]
        --> Controller[Flight Controller]
        --> ActuatorModel[Actuator Dynamics]
        --> FlowSolver[CFD Solver]
        --> Sensors[Virtual Sensors]
        --> Controller