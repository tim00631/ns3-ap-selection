#ifndef NS3_AI_INTERFACE_H
#define NS3_AI_INTERFACE_H

#include "ns3/ns3-ai-module.h"

namespace ns3 {


struct Env
{
	double rssi_ap_0;
	double rssi_ap_1;
	double rssi_ap_2;
	double rssi_ap_3;
	double rssi_ap_4;
	double rssi_ap_5;
	double rssi_ap_6;
	double rssi_ap_7;
	double rssi_ap_8;
	double reward;

} Packed;

struct Act
{
	uint32_t action;
} Packed;

class Ns3AI : public Ns3AIRL<Env, Act>
{
public:
	Ns3AI(uint16_t id);
	uint32_t step(std::vector<double> target_signalVec, double throughput);
};

} // namespace ns3

#endif // NS3_AI_INTERFACE_H