#include "ns3/command-line.h"
#include "ns3/double.h"
#include "ns3/string.h"
#include "ns3/ssid.h"

#include "ns3/yans-wifi-helper.h"
#include "ns3/mobility-helper.h"
#include "ns3/internet-stack-helper.h"
#include "ns3/ipv4-address-helper.h"
#include "ns3/flow-monitor-helper.h"
#include "ns3/ipv4-flow-classifier.h"

#include "ns3/csma-helper.h"
#include "ns3/bridge-helper.h"

#include "ns3/core-module.h"
#include "ns3/config-store-module.h"
#include "ns3/wifi-module.h"
#include "ns3/yans-wifi-channel.h"
#include "ns3/animation-interface.h"
#include "ns3/netanim-module.h"
#include "ns3/rng-seed-manager.h"
#include "ns3/applications-module.h"
#include "ns3/internet-module.h"
#include "ns3/ns3-ai-module.h"
#include "ns3-ai-interface.h"
#include <iomanip>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("APSelectionExperiment");

int memblock_key = 2333; /// < memory block key, need to keep the same in the python script
Ns3AI ns3AI(memblock_key);
double time_interval;
uint32_t recvBytes = 0;
uint32_t txBytes = 0;
double throughput = 0;
std::string cwd;
struct RssiMapEntry
{
	double signal_avg; // record 0.5s average SNR
	uint32_t n_samples;
};

std::unordered_map<std::string, uint32_t> AP_address_index_mapping;
std::vector<std::unordered_map<std::string, RssiMapEntry>> apVec;
std::vector<double> bg_signal_sumVec;
std::vector<double> target_signalVec;

static std::string ConvertMacAddressToStr (Mac48Address address) {
	std::stringstream stream;
	stream << address;
	return stream.str();
}

static uint32_t ConvertMacAddressToIndex (Mac48Address address) {
	std::stringstream stream;
	stream << address;
	auto it = AP_address_index_mapping.find(stream.str());
	if (it != AP_address_index_mapping.end()) {
		return it->second;
	}
	// NS_ASSERT(it != AP_address_index_mapping.end());
	else 
		return -1;
}

static std::string ConvertIndexToMacAddressStr (uint32_t i) {
	for (auto it : AP_address_index_mapping) {
		if(it.second == i) {
			return it.first;
		}
	}
	NS_LOG_INFO("NOT FOUND THIS INDEX\n");
	return "NOT FOUND THIS INDEX\n"; 
}

static Vector GetPosition (Ptr<Node> node) {
	Ptr<MobilityModel> mobility = node->GetObject<MobilityModel> ();
	return mobility->GetPosition();
}

static void PrintPositions (Ptr<Node> node)
{
  	// std::cout << "position: " << Simulator::Now ().GetMicroSeconds()/1000000.0 << " " << s << std::endl; 
    Ptr<MobilityModel> mob = node->GetObject<MobilityModel>();
    Vector pos = mob->GetPosition();
    std::cout  << "position: " << "  " << pos.x << ", " << pos.y << std::endl;
    Simulator::Schedule(Seconds(time_interval), (&PrintPositions), node);
}

void MonitorSniffRx (std::string context, 
					 Ptr<const Packet> packet,
                     uint16_t channelFreqMhz,
                     WifiTxVector txVector,
                     MpduInfo aMpdu,
                     SignalNoiseDbm signalNoise) {

	WifiMacHeader hdr;
	if(packet->PeekHeader(hdr)) {
		// if (hdr.IsProbeResp()) {
			
		if (hdr.IsBeacon()) {
			// hdr.Print(std::cout);
			// std::cout << std::endl;
			// std::cout << context << std::endl;
				// std::cout << "signal: " << signalNoise.signal << " ";

			std::string address = ConvertMacAddressToStr(hdr.GetAddr2()); // Beacon's AP address put in Addr2
			uint32_t index = ConvertMacAddressToIndex(hdr.GetAddr2());
			// std::cout << address << std::endl;
			// std::cout << "index: " << index << std::endl;

			// Vector version
			auto it_context = apVec[index].find(context);
			if (it_context != apVec[index].end()) {
				it_context->second.n_samples++;
				// it_context->second.signal_avg += ((signalNoise.signal-signalNoise.noise) - it_context->second.signal_avg) / it_context->second.n_samples;
				it_context->second.signal_avg += ((signalNoise.signal) - it_context->second.signal_avg) / it_context->second.n_samples;

			}
			else {
				RssiMapEntry entry;
				entry.n_samples = 1;
				// entry.signal_avg = signalNoise.signal-signalNoise.noise;
				entry.signal_avg = signalNoise.signal;

				apVec[index].insert(std::pair<std::string, RssiMapEntry>(context, entry));
			}
		}
	}
}

void GatherApInfo (Ptr<Node> targetStaNode) {
	double interference_th = -72;
	std::cout << "========================================================================" << std::endl;

	Ptr<WifiNetDevice> wifi_dev = DynamicCast<WifiNetDevice>(targetStaNode->GetDevice(0));
	Ptr<AiWifiMac> wifi_mac = DynamicCast<AiWifiMac>(wifi_dev->GetMac());
	// std::string address = ConvertMacAddressToStr(wifi_mac->GetAddress());

	for (uint32_t i = 0; i < apVec.size(); i++) {
		std::stringstream oss;
		oss << "/NodeList/" << targetStaNode->GetId() << "/DeviceList/0/Phy/MonitorSnifferRx";
		// std::cout << "sta: " << oss.str() << std::endl;
		std::string context = oss.str();
		for (auto it_sta: apVec[i]) {
			// std::cout << it_sta.first << std::endl;
			if (it_sta.first == context) {
				// std::cout << "skip self signal" << std::endl;
				target_signalVec[i] = it_sta.second.signal_avg;
			}
			else if (it_sta.second.signal_avg > interference_th) {
				bg_signal_sumVec[i] += it_sta.second.signal_avg;
			}
			else {
				std::stringstream oss;
				oss << "not to sum up " << it_sta.second.signal_avg  << " " << ConvertIndexToMacAddressStr(i) << std::endl;
				NS_LOG_DEBUG(oss.str());
			}
			// clear stats signal for every 0.5s
			it_sta.second.signal_avg = 0;
			it_sta.second.n_samples = 0;
		}
	}
	// Calculate SINR of target sta node
	std::stringstream oss;
	oss << "time " << std::setw(4) << Simulator::Now().GetSeconds() << "s RSSI:\n";
	for (uint32_t i = 0; i < target_signalVec.size(); i++) {
		oss << std::setw(10) << target_signalVec[i] << " ";
		target_signalVec[i] /= (bg_signal_sumVec[i]+ 1);
	}
	oss << std::endl;
	std::cout << oss.str();
	
	// Do AI Thing here 
	int action_i = ns3AI.step(target_signalVec, throughput);
	std::string s = ConvertIndexToMacAddressStr(action_i);
	Mac48Address bssid(s.c_str());
	if (wifi_mac->GetBssid() != bssid) {
		wifi_mac->SetNewAssociation(bssid);
	}

	// clear state vector when AI action done
	for (uint32_t i = 0; i < target_signalVec.size(); i++) {
		target_signalVec[i] = 0;
		bg_signal_sumVec[i] = 0;
	}
	// if (wifi_mac->IsAssociated()) {
	// 	std::cout << "associated to AP " << ConvertMacAddressToIndex(wifi_mac->GetBssid()) << ", ";
	// 	std::cout << "address: " << wifi_mac->GetBssid() << std::endl;
	// }
	// else {
	// 	std::cout << "Not associated...\n";
	// }
	Simulator::Schedule(Seconds(time_interval), (&GatherApInfo), targetStaNode);
}

void WhenAssociated(Mac48Address address) {

	std::cout << Simulator::Now().GetSeconds() << "s, associated to AP " << ConvertMacAddressToIndex(address) << ", addr:" << address << std::endl;
}

void CheckThroughput (FlowMonitorHelper* fmhelper, Ptr<FlowMonitor> flowMon, Ipv4InterfaceContainer* targetStaInterface)
{	
	// std::cout << "CheckThroughput " << Simulator::Now().GetSeconds() << std::endl;
	Ipv4Address targetStaAddr = targetStaInterface->GetAddress(0);
	flowMon->CheckForLostPackets(); 
	std::map<FlowId, FlowMonitor::FlowStats> flowStats = flowMon->GetFlowStats();
	Ptr<Ipv4FlowClassifier> classing = DynamicCast<Ipv4FlowClassifier> (fmhelper->GetClassifier());
	for (std::map<FlowId, FlowMonitor::FlowStats>::const_iterator stats = flowStats.begin (); stats != flowStats.end (); ++stats)
	{	
		Ipv4FlowClassifier::FiveTuple fiveTuple = classing->FindFlow (stats->first);
		if (fiveTuple.destinationAddress == targetStaAddr) {

		// 	// std::cout<<"Flow ID			: " << stats->first <<" ; "<< fiveTuple.sourceAddress <<" -----> "<<fiveTuple.destinationAddress<<std::endl;
		// 	// std::cout<<"Tx Packets = " << stats->second.txPackets<<std::endl;
		// 	// std::cout<<"Rx Packets = " << stats->second.rxPackets<<std::endl;
		// 	// std::cout<<"Duration		: "<<stats->second.timeLastRxPacket.GetSeconds()-stats->second.timeFirstTxPacket.GetSeconds()<<std::endl;
		// 	// std::cout<<"Last Received Packet	: "<< stats->second.timeLastRxPacket.GetSeconds()<<" Seconds"<<std::endl;
		// 	// std::cout<<"Throughput: " << stats->second.rxBytes * 8.0 / (stats->second.timeLastRxPacket.GetSeconds()-stats->second.timeFirstTxPacket.GetSeconds())/1024  << " kbps"<<std::endl;
			
			std::cout << "rxThroughput: " << (stats->second.rxBytes - recvBytes) * 8.0 / time_interval / 1024 / 1024 << " Mbps" << std::endl;
			std::cout << "TxThroughput: " << (stats->second.txBytes - txBytes) * 8.0 / time_interval / 1024 / 1024 << " Mbps" << std::endl;

			throughput = (stats->second.rxBytes - recvBytes) * 8.0 / time_interval / 1024 / 1024;
			recvBytes = stats->second.rxBytes;
			txBytes = stats->second.txBytes;
		}
		// std::cout<<"Flow ID			: " << stats->first <<" ; "<< fiveTuple.sourceAddress <<" -----> "<<fiveTuple.destinationAddress<<std::endl;
		// std::cout<<"Tx Packets = " << stats->second.txPackets<<std::endl;
		// std::cout<<"Rx Packets = " << stats->second.rxPackets<<std::endl;
		// std::cout<<"Duration		: "<<stats->second.timeLastRxPacket.GetSeconds()-stats->second.timeFirstTxPacket.GetSeconds()<<std::endl;
		// std::cout<<"Last Received Packet	: "<< stats->second.timeLastRxPacket.GetSeconds()<<" Seconds"<<std::endl;
		// std::cout<<"Throughput: " << stats->second.rxBytes * 8.0 / (stats->second.timeLastRxPacket.GetSeconds()-stats->second.timeFirstTxPacket.GetSeconds())/1024/1024  << " Mbps"<<std::endl;
		// std::cout << "---------------------------------------------------------------------------" << std::endl;

	}	
	Simulator::Schedule(Seconds(time_interval),&CheckThroughput, fmhelper, flowMon, targetStaInterface);
}

class APSelectionExperiment
{
public:
  	APSelectionExperiment();
  	void RunExperiment(uint32_t total_time,
       		 uint32_t nWifis,
       		 uint32_t nStas,
       		 double nodeSpeed,
       		 double nodePause,
       		 bool verbose,
			 bool enablePcap,
			 double txGain,
			 double rxGain,
			 double cca_edthreshold,
			 double txPower,
			 double exponent,
			 double referenceDistance);
private:
	void CreateNodes();
	void InstallSwitchLanDevices();
	void InstallWlanDevices();
	void InstallInternetStack();
	void InstallApplications();
	void SetAPMobility();
	void SetStaMobilityWithAPPosition(Vector ap_position, NodeContainer inRangeStas);
	void SetTargetStaMobility();
	static void CourseChange(std::string context, Ptr<const MobilityModel> model);

	uint32_t m_total_time;
	uint32_t m_nWifis;
	uint32_t m_nStas;
	double m_nodeSpeed;
	double m_nodePause;
	bool m_verbose;
	bool m_enablePcap;
	double m_txGain;
	double m_rxGain;
	double m_cca_edthreshold;
	double m_txPower;
	double m_exponent;
	double m_referenceDistance;

	// Nodes
	Ptr<Node> serverNode;
	Ptr<Node> metricServerNode;
	Ptr<Node> switchNode;
	Ptr<Node> targetStaNode;
	NodeContainer apNodes;
	NodeContainer staNodes;

	// Devices
	NetDeviceContainer switchDevices;
	NetDeviceContainer apEthDevices;
	NetDeviceContainer staDevices;
	NetDeviceContainer apDevices;
	Ptr<NetDevice> serverDevice;
	Ptr<NetDevice> metricServerDevice;
	Ptr<NetDevice> targetStaDevice;

	// Interfaces
	Ipv4InterfaceContainer staInterfaces;
	Ipv4InterfaceContainer apInterfaces;
	Ipv4InterfaceContainer serverInterface;
	Ipv4InterfaceContainer metricServerInterface;
	Ipv4InterfaceContainer targetStaInterface;
	Ipv4InterfaceContainer backBoneInterface;

};

APSelectionExperiment::APSelectionExperiment()
    : m_total_time(200000),
      m_nWifis(9),
      m_nStas(15),
      m_nodeSpeed(0.4),
      m_nodePause(0),
      m_verbose(false),
	  m_enablePcap(false),
	  m_txGain(5),
	  m_rxGain(5),
	  m_cca_edthreshold(-62),
	  m_txPower(21),
	  m_exponent(3),
	  m_referenceDistance(1)
{
  NS_LOG_FUNCTION(this);
}

void APSelectionExperiment::RunExperiment(uint32_t total_time,
                          uint32_t nWifis,
                          uint32_t nStas, 
                          double nodeSpeed, 
                          double nodePause,
                          bool verbose,
						  bool enablePcap,
						  double txGain,
						  double rxGain,
						  double cca_edthreshold,
						  double txPower,
						  double exponent,
			 			  double referenceDistance)
{
	m_total_time = total_time;
	m_nWifis = nWifis;
	m_nStas = nStas;
	m_nodeSpeed = nodeSpeed;
	m_nodePause = nodePause;
	m_verbose = verbose;
	m_enablePcap = enablePcap;
	m_txGain = txGain;
	m_rxGain = rxGain;
	m_cca_edthreshold = cca_edthreshold;
	m_txPower = txPower;
	m_exponent = exponent;
	m_referenceDistance = referenceDistance;

	CreateNodes();
	InstallSwitchLanDevices();
	InstallWlanDevices();
	InstallInternetStack();
	InstallApplications();

	// initialize the AP index map
	NS_LOG_INFO("initialize the AP index map");
	for (uint32_t i = 0; i < apNodes.GetN(); i++) {
		Ptr<WifiNetDevice> wifi_dev = DynamicCast<WifiNetDevice>(apNodes.Get(i)->GetDevice(1));
		Ptr<ApWifiMac> wifi_mac = DynamicCast<ApWifiMac>(wifi_dev->GetMac());
		std::stringstream address;
		address << wifi_mac->GetAddress();
		std::cout << address.str() << std::endl;
		AP_address_index_mapping.insert(std::pair<std::string, uint32_t>(address.str(), i));
		std::unordered_map<std::string, RssiMapEntry> staSnrTable;
		apVec.push_back(staSnrTable);
		bg_signal_sumVec.push_back(0);
		target_signalVec.push_back(0);
	}
	NS_ASSERT(AP_address_index_mapping.size() == m_nWifis);
	NS_ASSERT(target_signalVec.size() == m_nWifis);
	NS_ASSERT(bg_signal_sumVec.size() == m_nWifis);
	std::ostringstream ossta;
	ossta << "/NodeList/" << targetStaNode->GetId() << "/DeviceList/0/$ns3::WifiNetDevice/Mac/$ns3::AiWifiMac/Assoc";
	Config::ConnectWithoutContext(ossta.str(), MakeCallback(&WhenAssociated));

	// AsciiTraceHelper ascii;
	// MobilityHelper::EnableAsciiAll(ascii.CreateFileStream("wifi-wired-bridging.mob"));
	// Config::ConnectWithoutContext (oss.str(), MakeCallback (MakeCallback(&APSelectionExperiment::MonitorSniffRx));
	for (uint32_t i = 0; i < staNodes.GetN(); i++) {
		std::ostringstream oss;
		oss << "/NodeList/" << staNodes.Get(i)->GetId() << "/DeviceList/0/Phy/MonitorSnifferRx";
		Config::Connect(oss.str(), MakeCallback(&MonitorSniffRx));
	}
	std::ostringstream oss;
	oss << "/NodeList/" << targetStaNode->GetId() << "/DeviceList/0/Phy/MonitorSnifferRx";
	Config::Connect(oss.str(), MakeCallback(&MonitorSniffRx));

	AnimationInterface anim(cwd+"/rl-ap-selection-anim.xml");
	anim.SetMaxPktsPerTraceFile(10000000);
	anim.SetConstantPosition(switchNode, 0, 0, 0);
	anim.SetConstantPosition(serverNode, 0, 0, 0);
	anim.SetConstantPosition(metricServerNode, 0, 0, 0);
	if (m_enablePcap) {
		anim.EnablePacketMetadata(true);
	}
	// // Trace routing tables 
	// Ipv4GlobalRoutingHelper g;
	// Ptr<OutputStreamWrapper> routingStream = Create<OutputStreamWrapper> ("routes.txt", std::ios::out);
	// g.PrintRoutingTableAllAt (Seconds (4.0), routingStream);
	// print config
	// Config::SetDefault("ns3::ConfigStore::Filename", StringValue("output-attributes.txt"));
	// Config::SetDefault("ns3::ConfigStore::FileFormat", StringValue("RawText"));
	// Config::SetDefault("ns3::ConfigStore::Mode", StringValue("Save"));
	// ConfigStore outputConfig2;
	// outputConfig2.ConfigureDefaults();
	// outputConfig2.ConfigureAttributes();

	FlowMonitorHelper fmHelper;
	Ptr<FlowMonitor> monitor = fmHelper.Install(NodeContainer(targetStaNode, metricServerNode));
	Simulator::Schedule (Seconds (0 + time_interval), &CheckThroughput, &fmHelper, monitor, &targetStaInterface); 
	Simulator::Schedule(Seconds(0 + time_interval), (&GatherApInfo), targetStaNode);
	Simulator::Schedule(Seconds(0 + time_interval), (&PrintPositions), targetStaNode);

	Simulator::Stop(Seconds(m_total_time));
	Simulator::Run();
	Simulator::Destroy();
}

void APSelectionExperiment::CreateNodes()
{
	apNodes.Create(m_nWifis);
	switchNode = CreateObject<Node>();
	serverNode = CreateObject<Node>();
	targetStaNode = CreateObject<Node>();
	metricServerNode = CreateObject<Node>();

}

void APSelectionExperiment::SetAPMobility()
{
	NS_LOG_FUNCTION (this);
	NS_LOG_UNCOND ("SetAPMobility");
	/* Set Mobility for APs */
	MobilityHelper mobility;
	mobility.SetPositionAllocator("ns3::GridPositionAllocator",
								  "MinX", DoubleValue(50.0),
								  "MinY", DoubleValue(50.0), 
								  "DeltaX", DoubleValue(50.0), 
								  "DeltaY", DoubleValue(50.0),
								  "GridWidth", UintegerValue(3), 
								  "LayoutType", StringValue("RowFirst"));
	mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
	mobility.Install(apNodes);
}

void APSelectionExperiment::SetStaMobilityWithAPPosition(Vector ap_position, NodeContainer inRangeStas) {
	MobilityHelper mobility;
	/* Set Mobility for Stas */
	mobility.SetPositionAllocator(
		"ns3::UniformDiscPositionAllocator", 
		"X", StringValue(std::to_string(ap_position.x)), 
		"Y", StringValue(std::to_string(ap_position.y)), 
		"rho", StringValue(std::to_string(3.0)));
	mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
	mobility.Install(inRangeStas);
}

void APSelectionExperiment::SetTargetStaMobility() {
	MobilityHelper mobility;
	/* Set Mobility for target Stas */
	ObjectFactory objectFactoryPos;
	objectFactoryPos.SetTypeId("ns3::RandomRectanglePositionAllocator");
	objectFactoryPos.Set("X", StringValue("ns3::UniformRandomVariable[Min=0.0|Max=200.0]"));
	objectFactoryPos.Set("Y", StringValue("ns3::UniformRandomVariable[Min=0.0|Max=200.0]"));
	Ptr<PositionAllocator> positionAllocator =
		objectFactoryPos.Create()->GetObject<PositionAllocator>();

	std::stringstream ssSpeed;
	ssSpeed << "ns3::UniformRandomVariable[Min=" << m_nodeSpeed << "|Max=" << m_nodeSpeed << "]";
	std::stringstream ssPause;
	ssPause << "ns3::ConstantRandomVariable[Constant=" << m_nodePause << "]";
	mobility.SetPositionAllocator(positionAllocator);
	mobility.SetMobilityModel("ns3::RandomWaypointMobilityModel", "Speed",
									StringValue(ssSpeed.str()), "Pause", StringValue(ssPause.str()),
									"PositionAllocator", PointerValue(positionAllocator));
	mobility.Install(targetStaNode);

}

void APSelectionExperiment::InstallSwitchLanDevices()
{
	NS_LOG_UNCOND ("InstallSwitchLanDevices");
	CsmaHelper csmaHelper;
	BridgeHelper bridgeHelper;

	/* connect all ap and server to a switch */
	for(uint32_t i = 0; i < apNodes.GetN(); i++) {
		NetDeviceContainer link = csmaHelper.Install(NodeContainer(apNodes.Get(i), switchNode));
		apEthDevices.Add(link.Get(0));
		switchDevices.Add(link.Get(1));
	}

	NetDeviceContainer serverlink = csmaHelper.Install(NodeContainer(serverNode, switchNode));
	serverDevice = serverlink.Get(0);
	switchDevices.Add(serverlink.Get(1));

	NetDeviceContainer metricServerlink = csmaHelper.Install(NodeContainer(metricServerNode, switchNode));
	metricServerDevice = metricServerlink.Get(0);
	switchDevices.Add(metricServerlink.Get(1));

	bridgeHelper.Install(switchNode, switchDevices);
	if (m_enablePcap) {
		std::string s(cwd);
		csmaHelper.EnablePcap(s+"/pcap/server-eth", serverDevice);
		csmaHelper.EnablePcap(s+"/pcap/metric_server-eth", metricServerDevice);	
	}

	SetAPMobility();
}

void APSelectionExperiment::InstallWlanDevices()
{
	NS_LOG_UNCOND ("InstallWlanDevices");

	WifiHelper wifi;
	wifi.SetStandard(WIFI_PHY_STANDARD_80211g);
	// wifi.SetStandard (WIFI_PHY_STANDARD_80211n_2_4GHZ);
	std::string phyMode("ErpOfdmRate54Mbps");
	// Fix non-unicast data rate to be the same as that of unicast
	Config::SetDefault("ns3::WifiRemoteStationManager::NonUnicastMode", StringValue(phyMode));
	wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager",
								"DataMode",StringValue(phyMode),
								"ControlMode",StringValue(phyMode));
	YansWifiPhyHelper wifiPhy = YansWifiPhyHelper::Default();
	wifiPhy.Set("TxGain", DoubleValue(m_txGain));
	wifiPhy.Set("RxGain", DoubleValue(m_rxGain));
	wifiPhy.Set("CcaEdThreshold", DoubleValue(m_cca_edthreshold));
	wifiPhy.Set("TxPowerEnd", DoubleValue(m_txPower));
	wifiPhy.Set("TxPowerStart", DoubleValue(m_txPower));

	wifiPhy.SetPcapDataLinkType(WifiPhyHelper::DLT_IEEE802_11_RADIO);
	YansWifiChannelHelper wifiChannel;
	wifiChannel.SetPropagationDelay("ns3::ConstantSpeedPropagationDelayModel");
	wifiChannel.AddPropagationLoss ("ns3::LogDistancePropagationLossModel",
									"Exponent", DoubleValue (m_exponent),
									"ReferenceDistance", DoubleValue (m_referenceDistance));
	// 								// "ReferenceLoss", DoubleValue (40.0953)); // convert from mininet 

	wifiPhy.SetChannel(wifiChannel.Create());
	// std::vector<uint32_t> apChannelNum {1, 6, 1, 11, 1, 11, 1, 6, 1};
	std::vector<uint32_t> apChannelNum {1, 1, 1, 1, 1, 1, 1, 1, 1}; // ns3 doesn't support multi-channel scanning
	std::vector<uint32_t> n_bg_stas {0, 2, 3, 1, 0, 1, 3, 1, 1}; // Randomly create 0~3 background stas around this ap.

	WifiMacHelper wifiMac;
	Ssid ssid = Ssid("wifi-ssid");
	for (uint32_t i = 0; i < m_nWifis; ++i)
	{
		wifiPhy.Set("ChannelNumber", UintegerValue(apChannelNum[i]));
		wifiMac.SetType("ns3::ApWifiMac",
						"Ssid", SsidValue(ssid),
						"BeaconGeneration", BooleanValue(true));

		apDevices.Add(wifi.Install(wifiPhy, wifiMac, apNodes.Get(i)));
		BridgeHelper bridge;
		NetDeviceContainer bridgeDev;
      	bridgeDev = bridge.Install (apNodes.Get(i), 
                                  NetDeviceContainer (apDevices.Get(i), apEthDevices.Get(i))); // AP have two ports (wlan, eth)

		wifiMac.SetType("ns3::StaWifiMac",
					"Ssid", SsidValue(ssid),
					"ActiveProbing", BooleanValue(false));
		NodeContainer inRangeStas;
		inRangeStas.Create (n_bg_stas[i]);

		SetStaMobilityWithAPPosition(GetPosition(apNodes.Get(i)), inRangeStas);

		staNodes.Add(inRangeStas);
		staDevices.Add(wifi.Install(wifiPhy, wifiMac, inRangeStas));

	}

	wifiMac.SetType("ns3::AiWifiMac",
					"Ssid", SsidValue(ssid),
					"ActiveProbing", BooleanValue(false));	
	targetStaDevice = wifi.Install(wifiPhy, wifiMac, targetStaNode).Get(0);
	SetTargetStaMobility();
	if (m_enablePcap) {
		std::string s(cwd);
		wifiPhy.EnablePcap(s+"/pcap/ap-wlan", apDevices);
		wifiPhy.EnablePcap(s+"/pcap/sta-wlan", staDevices);
		wifiPhy.EnablePcap(s+"/pcap/target_sta-wlan", targetStaDevice);
	}
}

void APSelectionExperiment::InstallInternetStack() 
{
	NS_LOG_UNCOND ("InstallInternetStack");

	InternetStackHelper internet;
	Ipv4AddressHelper ip;
	
	internet.Install(staNodes);
	internet.Install(targetStaNode);
	internet.Install(apNodes);
	internet.Install(serverNode);
	internet.Install(metricServerNode);
	internet.Install(switchNode);
	
	ip.SetBase ("192.168.0.0", "255.255.255.0");
	serverInterface = ip.Assign(serverDevice); // 192.168.0.1
	metricServerInterface = ip.Assign(metricServerDevice); // 192.168.0.2
	staInterfaces = ip.Assign(staDevices); // 192.168.0.3 ~ 192.168.0.14
	targetStaInterface = ip.Assign(targetStaDevice); // 192.168.0.15
	
}

void APSelectionExperiment::InstallApplications()
{
	NS_LOG_UNCOND ("InstallApplications");
	// configure and install server/client app
	ApplicationContainer serverApps;
	ApplicationContainer clientApps;
	int serverPortBase = 8080;
	for(uint32_t i = 0; i < staNodes.GetN(); i++)
	{
		UdpEchoServerHelper server(serverPortBase);
		serverApps.Add(server.Install(serverNode));
		Address serverAddress = InetSocketAddress(serverInterface.GetAddress(0), serverPortBase++);

		UdpEchoClientHelper client(serverAddress);
		client.SetAttribute("MaxPackets", UintegerValue(4294967295u));
		client.SetAttribute("Interval", TimeValue(Seconds(0.005)));
		client.SetAttribute("PacketSize", UintegerValue(512));
		// std::cout << staInterfaces.GetAddress(i, 0) << std::endl;
		clientApps.Add(client.Install(staNodes.Get(i)));	
	}

	// UdpEchoServerHelper server(8000);
	// serverApps.Add(server.Install(serverNode));
	// Address serverAddress = InetSocketAddress(serverInterface.GetAddress(0), 8000);
	// UdpEchoClientHelper client1(serverAddress);
	// client1.SetAttribute("MaxPackets", UintegerValue(4294967295u));
	// client1.SetAttribute("Interval", TimeValue(Seconds(0.001)));
	// client1.SetAttribute("PacketSize", UintegerValue(512)); // Bytes
	// clientApps.Add(client1.Install(staNodes.Get(0)));


	UdpEchoServerHelper server2(9000);
	serverApps.Add(server2.Install(metricServerNode));
	Address serverAddress2 = InetSocketAddress(metricServerInterface.GetAddress(0), 9000);
	UdpEchoClientHelper client(serverAddress2);
	client.SetAttribute("MaxPackets", UintegerValue(4294967295u));
	client.SetAttribute("Interval", TimeValue(Seconds(0.001)));
	client.SetAttribute("PacketSize", UintegerValue(512)); // Bytes
	clientApps.Add(client.Install(targetStaNode));
	
	serverApps.Start(Seconds(0.0));
	serverApps.Stop(Seconds(m_total_time-0.5));

	clientApps.Start(Seconds(0.5));
	clientApps.Stop(Seconds(m_total_time));
}

void APSelectionExperiment::CourseChange(std::string context,Ptr<const MobilityModel> model)
{
  Vector position = model->GetPosition();
  NS_LOG_UNCOND("At time " << Simulator::Now().GetSeconds() << " " << context
                            << " x = " << position.x << ", y = " << position.y);
}
int
main(int argc, char *argv[])
{
	// LogComponentEnable ("StaWifiMac", LOG_FUNCTION);
	// LogComponentEnable ("UdpEchoClientApplication", LOG_INFO);
	// LogComponentEnable ("UdpEchoServerApplication", LOG_INFO);
	uint32_t total_time = 20000;
	uint32_t nWifis = 9;
	uint32_t nStas = 1;
	time_interval = 0.5;
	double nodeSpeed = 0.5; //in m/s	
	int nodePause = 0; //in s
	bool verbose = false;
	bool enablePcap = true;
	double txGain = 5;
	double rxGain = 5;
	double cca_edthreshold = -62;
	double txPower = 21.0;
	double exponent = 3.0;
	double referenceDistance = 1;

	// Control the Random Seed
	uint64_t seed = 123214242242;
	SeedManager::SetSeed(seed);
	SeedManager::SetRun(1);

	CommandLine cmd;
	cmd.AddValue("total_time", "Simulation time in seconds", total_time);
	cmd.AddValue("time_interval", "choose the action for every time interval", time_interval);
	cmd.AddValue("nWifis", "Number of wifi networks", nWifis);
	cmd.AddValue("nStas", "Number of stations", nStas);
	cmd.AddValue("enablePcap", "trace the pcap and push in /pcap dir", enablePcap);
	cmd.AddValue("verbose", "turn on all WifiNetDevice log components", verbose);
	cmd.AddValue("cwd", "Current working directory", cwd);
	cmd.Parse(argc, argv);
	std::cout << "nWifis:" << nWifis << ",nStas:" << nStas << std::endl;
	// std::cout << "plus:" << interface.step(nWifis, nStas) << std::endl;
	// Func can periodically call, and collect env info to python agent 
	APSelectionExperiment experiment = APSelectionExperiment();
	experiment.RunExperiment(total_time, nWifis, nStas, nodeSpeed, nodePause,
							 verbose, enablePcap, txGain, rxGain, cca_edthreshold,
							 txPower, exponent, referenceDistance);
	return 0;

}