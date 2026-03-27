#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/nr-module.h"
#include "ns3/applications-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/grid-scenario-helper.h"
#include <fstream>
#include <fcntl.h>
#include <unistd.h>

using namespace ns3;

static int      g_stateFd   = -1;
static int      g_actFd     = -1;
static float    g_sinr      = 0.0f;
static uint32_t g_beam      = 0;
static uint32_t g_prevBeam  = 0;
static uint32_t g_ttiCount  = 0;
static uint32_t g_staleCount= 0;
static std::ofstream g_diagLog;

void UpdateSinr(uint16_t, uint16_t,
                double sinr, uint16_t, uint8_t)
{
    g_sinr = (float)(10.0*std::log10(sinr+1e-9));
}

void TTIStep(Ptr<MobilityModel> mob)
{
    g_ttiCount++;
    Vector pos = mob->GetPosition();
    Vector vel = mob->GetVelocity();

    char buf[256];
    int n = snprintf(buf, sizeof(buf),
        "%.4f,%.4f,%.4f,%.4f,%.4f,%u\n",
        g_sinr,
        pos.x/200.0f, pos.y/200.0f,
        vel.x/5.0f,   vel.y/5.0f,
        g_prevBeam);
    write(g_stateFd, buf, n);

    char actBuf[32];
    int nr = read(g_actFd, actBuf,
                  sizeof(actBuf)-1);
    if (nr > 0) {
        actBuf[nr] = '\0';
        g_prevBeam = g_beam;
        g_beam     = (uint32_t)atoi(actBuf) % 32;
    } else {
        g_staleCount++;
    }

    Simulator::Schedule(MilliSeconds(1),
                        &TTIStep, mob);
}

int main(int argc, char* argv[])
{
    uint32_t    seed      = 1;
    double      simTime   = 10.0;
    double      speed     = 1.5;
    std::string statePipe = "/tmp/beam_state";
    std::string actPipe   = "/tmp/beam_act";

    CommandLine cmd;
    cmd.AddValue("seed",      "RNG seed",     seed);
    cmd.AddValue("simTime",   "Duration (s)", simTime);
    cmd.AddValue("speed",     "UE speed m/s", speed);
    cmd.AddValue("statePipe", "State pipe",   statePipe);
    cmd.AddValue("actPipe",   "Action pipe",  actPipe);
    cmd.Parse(argc, argv);

    RngSeedManager::SetSeed(seed);
    RngSeedManager::SetRun(seed);

    g_diagLog.open("diag_seed"+
                   std::to_string(seed)+".csv");
    g_diagLog << "time_ms,event,beam\n";

    g_stateFd = open(statePipe.c_str(), O_WRONLY);
    g_actFd   = open(actPipe.c_str(),
                     O_RDONLY|O_NONBLOCK);

    if (g_stateFd < 0 || g_actFd < 0) {
        std::cerr << "Failed to open pipes\n";
        return 1;
    }

    GridScenarioHelper grid;
    grid.SetRows(1);
    grid.SetColumns(1);
    grid.SetHorizontalBsDistance(5.0);
    grid.SetBsHeight(10.0);
    grid.SetUtHeight(1.5);
    grid.SetSectorization(
        GridScenarioHelper::SINGLE);
    grid.SetBsNumber(1);
    grid.SetUtNumber(1);
    grid.SetScenarioParameters(grid);
    grid.CreateScenario();

    Ptr<ConstantVelocityMobilityModel> cvMob =
        grid.GetUserTerminals().Get(0)
            ->GetObject<ConstantVelocityMobilityModel>();
    if (cvMob)
        cvMob->SetVelocity(
            Vector(speed,0.0,0.0));

    NodeContainer gnbNodes =
        grid.GetBaseStations();
    NodeContainer ueNodes  =
        grid.GetUserTerminals();

    Ptr<NrPointToPointEpcHelper> epcHelper =
        CreateObject<NrPointToPointEpcHelper>();
    Ptr<IdealBeamformingHelper> beamHelper =
        CreateObject<IdealBeamformingHelper>();
    Ptr<NrHelper> nrHelper =
        CreateObject<NrHelper>();

    nrHelper->SetBeamformingHelper(beamHelper);
    nrHelper->SetEpcHelper(epcHelper);
    beamHelper->SetBeamformingMethod(
        DirectPathBeamforming::GetTypeId());

    BandwidthPartInfoPtrVector allBwps;
    CcBwpCreator ccBwpCreator;
    CcBwpCreator::SimpleOperationBandConf bc(
        28e9, 100e6, 1,
        BandwidthPartInfo::UMi_StreetCanyon);
    OperationBandInfo band =
        ccBwpCreator
            .CreateOperationBandContiguousCc(bc);
    nrHelper->InitializeOperationBand(&band);
    allBwps = CcBwpCreator::GetAllBwps({band});

    nrHelper->SetGnbAntennaAttribute(
        "NumRows",    UintegerValue(4));
    nrHelper->SetGnbAntennaAttribute(
        "NumColumns", UintegerValue(4));
    nrHelper->SetUeAntennaAttribute(
        "NumRows",    UintegerValue(2));
    nrHelper->SetUeAntennaAttribute(
        "NumColumns", UintegerValue(2));
    nrHelper->SetGnbPhyAttribute(
        "TxPower", DoubleValue(43.0));

    NetDeviceContainer gnbDev =
        nrHelper->InstallGnbDevice(
            gnbNodes, allBwps);
    NetDeviceContainer ueDev  =
        nrHelper->InstallUeDevice(
            ueNodes, allBwps);

    for (auto it = gnbDev.Begin();
         it != gnbDev.End(); ++it)
        DynamicCast<NrGnbNetDevice>(*it)
            ->UpdateConfig();
    for (auto it = ueDev.Begin();
         it != ueDev.End(); ++it)
        DynamicCast<NrUeNetDevice>(*it)
            ->UpdateConfig();

    InternetStackHelper internet;
    internet.Install(ueNodes);
    epcHelper->AssignUeIpv4Address(ueDev);
    nrHelper->AttachToClosestEnb(ueDev,gnbDev);

    NodeContainer remote;
    remote.Create(1);
    internet.Install(remote);
    PointToPointHelper p2p;
    p2p.SetDeviceAttribute("DataRate",
        StringValue("10Gbps"));
    p2p.SetChannelAttribute("Delay",
        StringValue("1ms"));
    auto bd = p2p.Install(
        remote.Get(0),
        epcHelper->GetPgwNode());
    Ipv4AddressHelper addr;
    addr.SetBase("1.0.0.0","255.0.0.0");
    addr.Assign(bd);
    Ipv4StaticRoutingHelper routing;
    routing.GetStaticRouting(
        remote.Get(0)->GetObject<Ipv4>())
        ->AddNetworkRouteTo(
            Ipv4Address("7.0.0.0"),
            Ipv4Mask("255.0.0.0"),1);

    Ipv4Address ueIp =
        ueNodes.Get(0)->GetObject<Ipv4>()
               ->GetAddress(1,0).GetLocal();
    UdpClientHelper dlClient(ueIp,1234);
    dlClient.SetAttribute("Interval",
        TimeValue(MilliSeconds(1)));
    dlClient.SetAttribute("MaxPackets",
        UintegerValue(0xFFFFFFFF));
    dlClient.SetAttribute("PacketSize",
        UintegerValue(512));
    auto dlApps =
        dlClient.Install(remote.Get(0));
    dlApps.Start(Seconds(0.5));
    dlApps.Stop(Seconds(simTime));
    UdpServerHelper dlSrv(1234);
    dlSrv.Install(ueNodes.Get(0))
         .Start(Seconds(0.0));

    Config::ConnectWithoutContext(
        "/NodeList/*/DeviceList/*"
        "/$ns3::NrUeNetDevice"
        "/ComponentCarrierMapUe/*/NrUePhy"
        "/DlDataSinr",
        MakeCallback(&UpdateSinr));

    Ptr<MobilityModel> mob =
        ueNodes.Get(0)
               ->GetObject<MobilityModel>();
    Simulator::Schedule(MilliSeconds(500),
                        &TTIStep, mob);

    Simulator::Stop(Seconds(simTime));
    Simulator::Run();
    Simulator::Destroy();

    close(g_stateFd);
    close(g_actFd);
    g_diagLog.close();

    double sr = g_ttiCount > 0
        ? (double)g_staleCount/
          g_ttiCount*100.0 : 0.0;
    std::cout << "Done. TTIs=" << g_ttiCount
              << " StaleRate=" << sr << "%\n";
    return 0;
}
