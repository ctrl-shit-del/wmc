#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/applications-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/nr-module.h"
#include "ns3/antenna-module.h"
#include <fstream>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("BaselineBeamSweep");

static std::ofstream g_logFile;
static uint32_t g_currentBeamId = 0;

void
NotifySlotIndication(uint64_t frameNum, uint32_t sfNum,
                     uint32_t slotNum, uint32_t symStart,
                     uint32_t numSym, uint16_t bwpId)
{
    // slot tick — used to timestamp SINR logs
}

void
ReportSinr(uint16_t cellId, uint16_t rnti,
           double rsrp, double sinr, uint8_t componentCarrierId)
{
    double sinrDb = 10.0 * std::log10(sinr);
    double rsrpDbm = rsrp;
    g_logFile << Simulator::Now().GetMilliSeconds() << ","
              << sinrDb << ","
              << rsrpDbm << ","
              << g_currentBeamId << "\n";
}

int
main(int argc, char* argv[])
{
    uint32_t seed       = 1;
    uint32_t numBeams   = 32;
    double   simTimeSec = 10.0;
    double   ueSpeed    = 1.5;

    CommandLine cmd;
    cmd.AddValue("seed",    "RNG seed",        seed);
    cmd.AddValue("numBeams","Codebook size",    numBeams);
    cmd.AddValue("simTime", "Sim duration (s)", simTimeSec);
    cmd.AddValue("ueSpeed", "UE speed (m/s)",   ueSpeed);
    cmd.Parse(argc, argv);

    RngSeedManager::SetSeed(seed);
    RngSeedManager::SetRun(seed);

    std::string outFile = "baseline_seed" + std::to_string(seed) + ".csv";
    g_logFile.open(outFile);
    g_logFile << "time_ms,sinr_db,rsrp_dbm,beam_id\n";

    // ── Nodes ──────────────────────────────────────────────
    NodeContainer gnbNodes, ueNodes;
    gnbNodes.Create(1);
    ueNodes.Create(1);

    // ── Mobility ───────────────────────────────────────────
    // gNB fixed at origin, 10 m height
    MobilityHelper gnbMob;
    gnbMob.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    gnbMob.Install(gnbNodes);
    gnbNodes.Get(0)->GetObject<MobilityModel>()
            ->SetPosition(Vector(0.0, 0.0, 10.0));

    // UE: Random Waypoint inside 200x200 m cell, 1.5 m height
    MobilityHelper ueMob;
    ueMob.SetPositionAllocator(
        "ns3::RandomBoxPositionAllocator",
        "X", StringValue("ns3::UniformRandomVariable[Min=10|Max=190]"),
        "Y", StringValue("ns3::UniformRandomVariable[Min=10|Max=190]"),
        "Z", StringValue("ns3::ConstantRandomVariable[Constant=1.5]"));
    ueMob.SetMobilityModel(
        "ns3::RandomWaypointMobilityModel",
        "Speed",  StringValue(
            "ns3::ConstantRandomVariable[Constant=" +
            std::to_string(ueSpeed) + "]"),
        "Pause",  StringValue("ns3::ConstantRandomVariable[Constant=0]"),
        "PositionAllocator",
            StringValue("ns3::RandomBoxPositionAllocator"));
    ueMob.Install(ueNodes);

    // ── NR Helper setup ────────────────────────────────────
    Ptr<NrPointToPointEpcHelper> epcHelper =
        CreateObject<NrPointToPointEpcHelper>();
    Ptr<IdealBeamformingHelper> beamHelper =
        CreateObject<IdealBeamformingHelper>();
    Ptr<NrHelper> nrHelper = CreateObject<NrHelper>();

    nrHelper->SetBeamformingHelper(beamHelper);
    nrHelper->SetEpcHelper(epcHelper);

    // 28 GHz band, 100 MHz BW, UMi channel
    BandwidthPartInfo::Scenario scenario =
        BandwidthPartInfo::UMi_StreetCanyon;

    CcBwpCreator ccBwpCreator;
    const uint8_t numCcPerBand = 1;
    CcBwpCreator::SimpleOperationBandConf bandConf(
        28e9, 100e6, numCcPerBand, scenario);
    OperationBandInfo band =
        ccBwpCreator.CreateOperationBandContiguousCc(bandConf);

    nrHelper->InitializeOperationBand(&band);

    // gNB antenna: 4x4 array
    nrHelper->SetGnbAntennaAttribute("NumRows",    UintegerValue(4));
    nrHelper->SetGnbAntennaAttribute("NumColumns", UintegerValue(4));
    // UE antenna: 2x2 array
    nrHelper->SetUeAntennaAttribute("NumRows",    UintegerValue(2));
    nrHelper->SetUeAntennaAttribute("NumColumns", UintegerValue(2));

    // Beamforming: Cell scan (exhaustive sweep — P1 baseline)
    beamHelper->SetBeamformingMethod(
        CellScanBeamforming::GetTypeId());

    NetDeviceContainer gnbDev =
        nrHelper->InstallGnbDevice(gnbNodes, band.GetBwpVec());
    NetDeviceContainer ueDev  =
        nrHelper->InstallUeDevice(ueNodes,  band.GetBwpVec());

    // Internet stack
    InternetStackHelper internet;
    internet.Install(ueNodes);
    epcHelper->AssignUeIpv4Address(ueDev);
    nrHelper->AttachToClosestEnb(ueDev, gnbDev);

    // ── UDP traffic: gNB → UE downlink ────────────────────
    Ptr<Node> pgw = epcHelper->GetPgwNode();
    NodeContainer remoteHosts;
    remoteHosts.Create(1);
    internet.Install(remoteHosts);

    PointToPointHelper p2p;
    p2p.SetDeviceAttribute("DataRate",
                           StringValue("10Gbps"));
    p2p.SetChannelAttribute("Delay", StringValue("1ms"));
    NetDeviceContainer backbone =
        p2p.Install(remoteHosts.Get(0), pgw);

    Ipv4AddressHelper ipv4h;
    ipv4h.SetBase("1.0.0.0", "255.0.0.0");
    Ipv4InterfaceContainer backboneIfaces =
        ipv4h.Assign(backbone);

    Ipv4StaticRoutingHelper routingHelper;
    Ptr<Ipv4StaticRouting> remoteHostRouting =
        routingHelper.GetStaticRouting(
            remoteHosts.Get(0)->GetObject<Ipv4>());
    remoteHostRouting->AddNetworkRouteTo(
        Ipv4Address("7.0.0.0"), Ipv4Mask("255.0.0.0"), 1);

    Ipv4Address ueAddr =
        ueNodes.Get(0)->GetObject<Ipv4>()
               ->GetAddress(1, 0).GetLocal();

    uint16_t port = 1234;
    UdpClientHelper dlClient(ueAddr, port);
    dlClient.SetAttribute("Interval",
        TimeValue(MilliSeconds(1)));
    dlClient.SetAttribute("MaxPackets",
        UintegerValue(0xFFFFFFFF));
    dlClient.SetAttribute("PacketSize", UintegerValue(1000));

    ApplicationContainer dlApps =
        dlClient.Install(remoteHosts.Get(0));
    dlApps.Start(Seconds(0.5));
    dlApps.Stop(Seconds(simTimeSec));

    UdpServerHelper dlServer(port);
    dlServer.Install(ueNodes.Get(0))
            .Start(Seconds(0.0));

    // ── SINR trace ─────────────────────────────────────────
    Config::ConnectWithoutContext(
        "/NodeList/*/DeviceList/*/ComponentCarrierMapUe/*/NrUePhy"
        "/ReportCurrentCellRsrpSinr",
        MakeCallback(&ReportSinr));

    Simulator::Stop(Seconds(simTimeSec));
    Simulator::Run();
    Simulator::Destroy();

    g_logFile.close();
    std::cout << "Done. Results: " << outFile << std::endl;
    return 0;
}
