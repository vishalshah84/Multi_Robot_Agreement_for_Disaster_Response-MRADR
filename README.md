# Multi-Robot Agreement for Disaster Response (MRADR)

This project simulates a team of robots moving in a 2D city environment.  
Robots patrol the city, detect hazards, share information with nearby robots, reach agreement using distributed consensus, and move together to respond to the hazard.

The simulation is fully decentralized — no central controller.

---

## 🔧 How It Works (Short Summary)

- Robots continuously move around the city.
- A hazard appears at a random building.
- Robots detect hazards only when close.
- Robots exchange their beliefs with neighbors.
- Using **average consensus**, all robots agree on the correct hazard location.
- Once agreement is reached, all robots move to respond to that building.
- A new hazard appears and the cycle repeats.

---

## ▶️ How to Run

### **1. Install dependencies**

```bash
pip3 install pygame numpy
