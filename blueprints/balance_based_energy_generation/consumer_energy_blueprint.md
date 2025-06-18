# Consumer-Grade Energy Harvesting Prototype

## Objective

Design a low-cost, modular, and safe prototype for extracting energy from a carbon-based core using controlled electromagnetic pulses.

## 1. Core Material

- **Material**: Graphite (85–90% purity)

- **Form Factor**: Cylindrical core

  - Diameter: 5 cm

  - Height: 5.6 cm

  - Mass: 500 g

## 2. Pulse Induction Unit

- **High-voltage Capacitor**: 200–400 J range

  - Camera flash capacitors or equivalent

- **Pulse Induction Coil**:

  - Copper wire, 200–400 turns

  - Solenoid type, wrapped around the core

- **Microcontroller**: Raspberry Pi or Arduino

  - Controls pulse frequency and safety thresholds

## 3. Energy Harvesting System

### a. Electromagnetic

- Secondary copper coil (100–200 turns)

- Diameter: 3 cm, near the base of the core

### b. Thermal

- **TEGs**: 2–4 thermoelectric generator modules

- **Placement**: Around the graphite core

## 4. Energy Storage and Output Regulation

- **Storage**: Capacitor bank or Li-ion battery (1000 mAh)

- **Voltage Regulation**: 5V buck converter

## 5. Housing and Containment

- **Material**: Plastic, ceramic, or carbon fiber

- **Size**: 10 cm × 10 cm × 10 cm

- **Cooling**: Passive (vents) or active (small fans)

## 6. Safety and Control

- **Overcurrent Protection**: Fuse or thermal breaker

- **EM Shielding**: Protect microcontroller from pulses

## Expected Output

- ~200–700 J per pulse

- Usable output: ~100–200 J (with ~30% harvesting efficiency)

## Use Cases

- Portable power (phones, LEDs, fans)

- Emergency backup or off-grid energy

## Next Steps

- 3D print housing and test with graphite core

- Optimize pulse strength, coil design, and TEG placement
