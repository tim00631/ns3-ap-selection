#ifndef NS3_AI_INTERFACE_H
#define NS3_AI_INTERFACE_H

#include "ns3/ns3-ai-module.h"

namespace ns3 {


struct Env
{
	int action;
	double reward;

} Packed;

struct Act
{
	uint32_t empty;
} Packed;

class Ns3AI : public Ns3AIRL<Env, Act>
{
public:
	Ns3AI(uint16_t id);
	uint32_t step(int action, double throughput);
};

} // namespace ns3

#endif // NS3_AI_INTERFACE_H