#include "ns3/command-line.h"
#include "ns3/double.h"
#include "ns3/string.h"
#include "ns3/ssid.h"

#include "ns3/yans-wifi-helper.h"
#include "ns3/mobility-helper.h"
#include "ns3/internet-stack-helper.h"
#include "ns3/ipv4-address-helper.h"
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

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("APSelectionExperiment");
uint32_t nWifis = 9;

struct RssiMapEntry
{
	double signal_avg; // record 0.5s average SNR
	uint32_t n_samples;
};
std::unordered_map<std::string, std::unordered_map<std::string, RssiMapEntry>> apTable; // every AP should maintain a Map which is formed as <STA context, SignaldBm>;
std::unordered_map<std::string, double> bg_signal_sum; // record the summation of signal
std::unordered_map<std::string, double> target_signal; // record target STA signal;
static Vector GetPosition (Ptr<Node> node) {
	Ptr<MobilityModel> mobility = node->GetObject<MobilityModel> ();
	return mobility->GetPosition();
}

static void PrintPositions(std::string s, Ptr<Node> node)
{
  	std::cout << "t = " << Simulator::Now ().GetMicroSeconds()/1000000.0 << " " << s << std::endl; 
    Ptr<MobilityModel> mob = node->GetObject<MobilityModel>();
    Vector pos = mob->GetPosition ();
    std::cout  << "position: " << "  " << pos.x << ", " << pos.y << std::endl;
    Simulator::Schedule(Seconds(1), (&PrintPositions), s, node);
}

void GatherApInfo (Ptr<Node> targetStaNode) {
	double interference_th = -72;
	std::cout << "GatherApInfo" << std::endl;
	for (auto it_ap: apTable) { // for every AP, summate their neighbor STA's SNR
		double sum = 0;
		for (auto it_sta: it_ap.second) {
			Ptr<WifiNetDevice> wifi_dev = DynamicCast<WifiNetDevice>(targetStaNode->GetDevice(0));
			Ptr<StaWifiMac> wifi_mac = DynamicCast<StaWifiMac>(wifi_dev->GetMac());
			std::stringstream address;
			address << wifi_mac->GetAddress();
			if (it_sta.first == address.str()) {
				std::cout << "skip self signal" << std::endl;
				target_signal.insert(std::pair<std::string, double>(it_ap.first, it_sta.second.signal_avg));
			}
			else if (it_sta.second.signal_avg > interference_th) {
				sum += it_sta.second.signal_avg;
			}
			else {
				std::cout << "not to sum up " << it_sta.second.signal_avg  << " " << it_ap.first << std::endl;
			}
			it_sta.second.signal_avg = 0;
			it_sta.second.n_samples = 0;
		}
		std::cout << it_ap.first << " sum:" << sum << std::endl;
		bg_signal_sum.insert(std::pair<std::string, double>(it_ap.first, sum));
	}
	/*
	* SINR caluation S/(I+N)
	*/
	Simulator::Schedule(Seconds(0.5), (&GatherApInfo), targetStaNode);
}

static std::string ConvertMacAddressToStr (Mac48Address address) {
	std::stringstream stream;
	stream << address;
	return stream.str();
}

void MonitorSniffRx (std::string context, 
					 Ptr<const Packet> packet,
                     uint16_t channelFreqMhz,
                     WifiTxVector txVector,
                     MpduInfo aMpdu,
                     SignalNoiseDbm signalNoise) {
	// g_samples++;
	// g_signalDbmAvg = signalNoise.signal;
	// g_noiseDbmAvg = signalNoise.noise;
	WifiMacHeader hdr;
	if(packet->PeekHeader(hdr)) {
		if (hdr.IsBeacon()) {
			// hdr.Print(std::cout);
			// std::cout << std::endl;
			// std::cout << context << std::endl;
			std::cout << "signal: " << signalNoise.signal << " ";
			// std::cout << "\t(Beacon RA) Addr2: " << hdr.GetAddr2() << " " << std::endl; // Beacon's AP address put in Addr2
			// std::cout << "\t(Beacon RA) Addr3: " << hdr.GetAddr3() << " " << std::endl;

			std::string address = ConvertMacAddressToStr(hdr.GetAddr2());
			std::cout << address << std::endl;
			auto it_ApTable = apTable.find(address);
			if (it_ApTable != apTable.end()) {
				auto it_staSnrTable = it_ApTable->second.find(context);
				if (it_staSnrTable != it_ApTable->second.end()) {
					it_staSnrTable->second.n_samples++;
					it_staSnrTable->second.signal_avg += (signalNoise.signal - it_staSnrTable->second.signal_avg) / it_staSnrTable->second.n_samples;
				}
				else {
					RssiMapEntry entry;
					entry.n_samples = 1;
					entry.signal_avg = signalNoise.signal;
					it_ApTable->second.insert(std::pair<std::string, RssiMapEntry>(context, entry));
				}
			}
			else {
				std::unordered_map<std::string, RssiMapEntry> staSnrTable;
				apTable.insert(std::pair<std::string, std::unordered_map<std::string, RssiMapEntry>>(address, staSnrTable));
			}
		}
	}
}

struct Env
{
  int a;
  int b;
} Packed;

struct Act
{
  int c;
} Packed;

class APB : public Ns3AIRL<Env, Act>
{
public:
  APB(uint16_t id);
  int Func(int a, int b);
};

/**
 * \brief Link the shared memory with the id and set the operation lock
 *
 * \param[in] id  shared memory id, should be the same in python and ns-3
 */
APB::APB(uint16_t id) : Ns3AIRL<Env, Act>(id)
{
  SetCond(2, 0); ///< Set the operation lock(even for ns-3 and odd for python).
}

int
APB::Func(int a, int b)
{
  auto env = EnvSetterCond(); ///< Acquire the Env memory for writing
  env->a = a;
  env->b = b;
  NS_LOG_UNCOND("sim.cc Set Env:a=" << env->a << ", b=" << env->b << std::endl);
  SetCompleted(); ///< Release the memory and update conters
  NS_LOG_DEBUG("Ver:" << (int) SharedMemoryPool::Get()->GetMemoryVersion(m_id));
  auto act = ActionGetterCond(); ///< Acquire the Act memory for reading
  int ret = act->c;
  NS_LOG_UNCOND("sim.cc Get Act:c=" << act->c << std::endl);

  GetCompleted(); ///< Release the memory, roll back memory version and update conters
  NS_LOG_DEBUG("Ver:" << (int) SharedMemoryPool::Get()->GetMemoryVersion(m_id));
  return ret;
}

class APSelectionExperiment
{
public:
  	APSelectionExperiment();
  	void RunExperiment(uint32_t totalTime,
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
	void CheckThroughput();
	static void CourseChange(std::string context, Ptr<const MobilityModel> model);

	uint32_t m_totalTime;
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
	Ptr<NetDevice> targetStaDevice;

	// Interfaces
	Ipv4InterfaceContainer staInterfaces;
	Ipv4InterfaceContainer apInterfaces;
	Ipv4InterfaceContainer serverInterface;
	Ipv4InterfaceContainer targetStaInterface;
	Ipv4InterfaceContainer backBoneInterface;

};

APSelectionExperiment::APSelectionExperiment()
    : m_totalTime(200000),
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

void APSelectionExperiment::RunExperiment(uint32_t totalTime,
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
	m_totalTime = totalTime;
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

	// std::ostringstream ossta;
	// ossta << "/NodeList/" << targetStaNode->GetId() << "/$ns3::MobilityModel/CourseChange";
	// Config::Connect(ossta.str(), MakeCallback(&APSelectionExperiment::CourseChange));
	// AsciiTraceHelper ascii;
	// MobilityHelper::EnableAsciiAll(ascii.CreateFileStream("wifi-wired-bridging.mob"));
	std::ostringstream oss;
	oss << "/NodeList/*/DeviceList/*/Phy/MonitorSnifferRx";
	// Config::ConnectWithoutContext (oss.str(), MakeCallback (MakeCallback(&APSelectionExperiment::MonitorSniffRx));
	Config::Connect(oss.str(), MakeCallback(&MonitorSniffRx));
	AnimationInterface anim("/home/hscc/ns-3-allinone/ns-3.30/scratch/rl-ap-selection/rl-ap-selection-anim.xml");
	anim.SetMaxPktsPerTraceFile(10000000);
	anim.SetConstantPosition(switchNode, 0, 0, 0);
	anim.SetConstantPosition(serverNode, 0, 0, 0);
	if (m_enablePcap) {
		// anim.EnablePacketMetadata(true);
	}
	// // Trace routing tables 
	// Ipv4GlobalRoutingHelper g;
	// Ptr<OutputStreamWrapper> routingStream = Create<OutputStreamWrapper> ("/home/hscc/ns-3-allinone/ns-3.30/scratch/rl-ap-selection/routes.txt", std::ios::out);
	// g.PrintRoutingTableAllAt (Seconds (4.0), routingStream);
	// print config
	// Config::SetDefault("ns3::ConfigStore::Filename", StringValue("/home/hscc/ns-3-allinone/ns-3.30/scratch/rl-ap-selection/output-attributes.txt"));
	// Config::SetDefault("ns3::ConfigStore::FileFormat", StringValue("RawText"));
	// Config::SetDefault("ns3::ConfigStore::Mode", StringValue("Save"));
	// ConfigStore outputConfig2;
	// outputConfig2.ConfigureDefaults();
	// outputConfig2.ConfigureAttributes();

	// Simulator::Schedule(Seconds(0), (&PrintPositions), "target-sta", targetStaNode);
	// PacketMetadata::Enable();
	// Packet::EnableChecking();
	// Packet::EnablePrinting();
	Simulator::Schedule(Seconds(0), (&GatherApInfo), targetStaNode);
	Simulator::Stop(Seconds(m_totalTime));
	Simulator::Run();
	Simulator::Destroy();
}

void APSelectionExperiment::CreateNodes()
{
	apNodes.Create(m_nWifis);
	// staNodes.Create(m_nStas);
	switchNode = CreateObject<Node>();
	serverNode = CreateObject<Node>();
	targetStaNode = CreateObject<Node>();

}

void APSelectionExperiment::SetAPMobility()
{
	NS_LOG_FUNCTION (this);
	NS_LOG_UNCOND ("SetAPMobility");
	/* Set Mobility for APs */
	MobilityHelper mobility;
	mobility.SetPositionAllocator("ns3::GridPositionAllocator",
								  "MinX", DoubleValue(10.0),
								  "MinY", DoubleValue(10.0), 
								  "DeltaX", DoubleValue(10.0), 
								  "DeltaY", DoubleValue(10.0),
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
		"rho", StringValue(std::to_string(5.0)));
	mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
	mobility.Install(inRangeStas);
}

void APSelectionExperiment::SetTargetStaMobility() {
	MobilityHelper mobility;
	/* Set Mobility for target Stas */
	ObjectFactory objectFactoryPos;
	objectFactoryPos.SetTypeId("ns3::RandomRectanglePositionAllocator");
	objectFactoryPos.Set("X", StringValue("ns3::UniformRandomVariable[Min=0.0|Max=40.0]"));
	objectFactoryPos.Set("Y", StringValue("ns3::UniformRandomVariable[Min=0.0|Max=40.0]"));
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
	NetDeviceContainer link = csmaHelper.Install(NodeContainer(serverNode, switchNode));
	serverDevice = link.Get(0);
	switchDevices.Add(link.Get(1));

	bridgeHelper.Install(switchNode, switchDevices);
	if (m_enablePcap) {
		csmaHelper.EnablePcap("/home/hscc/ns-3-allinone/ns-3.30/scratch/rl-ap-selection/pcap/server-eth", serverDevice);	
	}

	SetAPMobility();
}

void APSelectionExperiment::InstallWlanDevices()
{
	NS_LOG_UNCOND ("InstallWlanDevices");

	WifiHelper wifi;
	wifi.SetStandard(WIFI_PHY_STANDARD_80211g);
	// wifi.SetStandard (WIFI_PHY_STANDARD_80211n_2_4GHZ);
	// std::string phyMode("DsssRate1Mbps");
	// Fix non-unicast data rate to be the same as that of unicast
	// Config::SetDefault("ns3::WifiRemoteStationManager::NonUnicastMode", StringValue(phyMode));
	// wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager",
	// 							"DataMode",StringValue(phyMode),
	// 							"ControlMode",StringValue(phyMode));
	YansWifiPhyHelper wifiPhy = YansWifiPhyHelper::Default();
	wifiPhy.Set("TxGain", DoubleValue(m_txGain));
	wifiPhy.Set("RxGain", DoubleValue(m_rxGain));
	wifiPhy.Set("CcaEdThreshold", DoubleValue(m_cca_edthreshold));
	wifiPhy.Set("TxPowerEnd", DoubleValue(m_txPower));
	wifiPhy.Set("TxPowerStart", DoubleValue(m_txPower));

	wifiPhy.SetPcapDataLinkType(WifiPhyHelper::DLT_IEEE802_11_RADIO);
	YansWifiChannelHelper wifiChannel = YansWifiChannelHelper::Default();
	wifiChannel.AddPropagationLoss ("ns3::LogDistancePropagationLossModel",
									"Exponent", DoubleValue (m_exponent),
									"ReferenceDistance", DoubleValue (m_referenceDistance),
									"ReferenceLoss", DoubleValue (40.0953)); // convert from mininet 

	wifiPhy.SetChannel(wifiChannel.Create());
	std::vector<uint32_t> apChannelNum {1, 6, 1, 11, 1, 11, 1, 6, 1};

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
		inRangeStas.Create (m_nStas);

		SetStaMobilityWithAPPosition(GetPosition(apNodes.Get(i)), inRangeStas);

		staNodes.Add(inRangeStas);
		staDevices.Add(wifi.Install(wifiPhy, wifiMac, inRangeStas));

	}

	wifiMac.SetType("ns3::StaWifiMac",
					"Ssid", SsidValue(ssid),
					"ActiveProbing", BooleanValue(false));	
	targetStaDevice = wifi.Install(wifiPhy, wifiMac, targetStaNode).Get(0);
	SetTargetStaMobility();
	if (m_enablePcap) {
		wifiPhy.EnablePcap("/home/hscc/ns-3-allinone/ns-3.30/scratch/rl-ap-selection/pcap/ap-wlan", apDevices);
		wifiPhy.EnablePcap("/home/hscc/ns-3-allinone/ns-3.30/scratch/rl-ap-selection/pcap/sta-wlan", staDevices);
		wifiPhy.EnablePcap("/home/hscc/ns-3-allinone/ns-3.30/scratch/rl-ap-selection/pcap/target_sta-wlan", targetStaDevice);
	}
}

void APSelectionExperiment::InstallInternetStack() 
{
	NS_LOG_UNCOND ("InstallInternetStack");
	// OlsrHelper olsr;
	// internet.SetRoutingHelper(olsr); // has effect on the next Install ()
	// internet.Install(apNodes);
	// internet.Install(serverNode);
	// internet.Install(staNodes);
	// internet.Install(targetStaNode);

	// ip.SetBase ("172.16.0.0", "255.255.255.0");
	// serverInterface = ip.Assign (apEthDevices.Get(m_nWifis));
	
	// Set IP
	// ip.SetBase ("172.16.0.0", "255.255.255.0");
	// staInterfaces = ip.Assign(staDevices);
	// targetStaInterface = ip.Assign(targetStaDevice);

	// Ipv4GlobalRoutingHelper::PopulateRoutingTables();

	InternetStackHelper internet;
	Ipv4AddressHelper ip;

	internet.Install(staNodes);
	internet.Install(targetStaNode);
	internet.Install(apNodes);
	internet.Install(serverNode);
	internet.Install(switchNode);
	
	ip.SetBase ("192.168.0.0", "255.255.255.0");
	serverInterface = ip.Assign(serverDevice);

	staInterfaces = ip.Assign(staDevices);
	targetStaInterface = ip.Assign(targetStaDevice);
}

void APSelectionExperiment::InstallApplications()
{
	NS_LOG_UNCOND ("InstallApplications");
	// configure and install server/client app
	int serverPortBase = 8080;
	ApplicationContainer serverApps;
	ApplicationContainer clientApps;
	for(uint32_t i = 0; i < staNodes.GetN(); i++)
	{
		UdpEchoServerHelper server(serverPortBase);
		serverApps.Add(server.Install(serverNode));
		Address serverAddress = InetSocketAddress(serverInterface.GetAddress(0), serverPortBase++);

		UdpEchoClientHelper client(serverAddress);
		client.SetAttribute("MaxPackets", UintegerValue(4294967295u));
		client.SetAttribute("Interval", TimeValue(Seconds(0.001)));
		client.SetAttribute("PacketSize", UintegerValue(512));
		clientApps.Add(client.Install(staNodes.Get(i)));
		
	}

	// UdpEchoServerHelper server2(9000);
	// serverApps.Add(server2.Install(serverNode));
	// std::cout << "serverInterface" << serverInterface.GetAddress(0) << std::endl;
	// Address serverAddress2 = InetSocketAddress(serverInterface.GetAddress(0), 9000);
	// UdpEchoClientHelper client(serverAddress2);
	// client.SetAttribute("MaxPackets", UintegerValue(4294967295u));
	// client.SetAttribute("Interval", TimeValue(Seconds(1)));
	// client.SetAttribute("PacketSize", UintegerValue(1024));
	// clientApps.Add(client.Install(staNodes.Get(0)));

	serverApps.Start(Seconds(1));
	serverApps.Stop(Seconds(m_totalTime+1));

	clientApps.Start(Seconds(2));
	clientApps.Stop(Seconds(m_totalTime+1));
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
	int memblock_key = 2333; ///< memory block key, need to keep the same in the python script
	// LogComponentEnable ("StaWifiMac", LOG_ALL);
	// LogComponentEnable ("UdpEchoClientApplication", LOG_INFO);
	// LogComponentEnable ("UdpEchoServerApplication", LOG_INFO);
	uint32_t totalTime = 10;
	nWifis = 9;
	uint32_t nStas = 1;
	double nodeSpeed = 0.4; //in m/s	
	int nodePause = 0; //in s
	bool verbose = false;
	bool enablePcap = true;
	double txGain = 5;
	double rxGain = 5;
	double cca_edthreshold = -62;
	double txPower = 21.0;
	double exponent = 3;
	double referenceDistance = 1;

	// Control the Random Seed
	uint64_t seed = 1239021930;
	SeedManager::SetSeed(seed);
	SeedManager::SetRun(1);

	CommandLine cmd;
	cmd.AddValue("totalTime", "Simulation time in seconds", totalTime);
	cmd.AddValue("nWifis", "Number of wifi networks", nWifis);
	cmd.AddValue("nStas", "Number of stations", nStas);
	cmd.AddValue("enablePcap", "trace the pcap and push in /pcap dir", enablePcap);
	cmd.AddValue("verbose", "turn on all WifiNetDevice log components", verbose);

	cmd.Parse(argc, argv);
	APB apb(memblock_key);
	std::cout << "nWifis:" << nWifis << ",nStas:" << nStas << std::endl;
	//   std::cout << "plus:" << apb.Func(nWifis, nStas) << std::endl;
	// Func can periodically call, and collect env info to python agent 
	APSelectionExperiment experiment = APSelectionExperiment();
	experiment.RunExperiment(totalTime, nWifis, nStas, nodeSpeed, nodePause,
							 verbose, enablePcap, txGain, rxGain, cca_edthreshold,
							 txPower, exponent, referenceDistance);
	return 0;

}