#include "ns3-ai-interface.h"
#include "ns3/log.h"


namespace ns3
{

NS_LOG_COMPONENT_DEFINE("ns3-ai-interface");

/**
 * \brief Link the shared memory with the id and set the operation lock
 *
 * \param[in] id  shared memory id, should be the same in python and ns-3
 */
Ns3AI::Ns3AI(uint16_t id) : Ns3AIRL<Env, Act>(id)
{
	SetCond(2, 0); ///< Set the operation lock(even for ns-3 and odd for python).
}

uint32_t
Ns3AI::step(std::vector<double> target_signalVec, double throughput)
{
	auto env = EnvSetterCond(); ///< Acquire the Env memory for writing
	env->rssi_ap_0 = target_signalVec[0];
	env->rssi_ap_1 = target_signalVec[1];
	env->rssi_ap_2 = target_signalVec[2];
	env->rssi_ap_3 = target_signalVec[3];
	env->rssi_ap_4 = target_signalVec[4];
	env->rssi_ap_5 = target_signalVec[5];
	env->rssi_ap_6 = target_signalVec[6];
	env->rssi_ap_7 = target_signalVec[7];
	env->rssi_ap_8 = target_signalVec[8];
	env->reward = throughput;
	SetCompleted(); ///< Release the memory and update conters
	NS_LOG_DEBUG("Ver:" << (int) SharedMemoryPool::Get()->GetMemoryVersion(m_id));
	auto act = ActionGetterCond(); ///< Acquire the Act memory for reading
	uint32_t action = act->action;
	GetCompleted(); ///< Release the memory, roll back memory version and update conters
	NS_LOG_DEBUG("Ver:" << (int) SharedMemoryPool::Get()->GetMemoryVersion(m_id));
	return action;
}

} // namespace ns3