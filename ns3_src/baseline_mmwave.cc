#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/nr-module.h"
#include "ns3/applications-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/nr-helper.h"
#include "ns3/nr-point-to-point-epc-helper.h"
#include "ns3/ideal-beamforming-helper.h"
#include "ns3/cc-bwp-helper.h"
#include "ns3/grid-scenario-helper.h"
#include <fstream>

using namespace ns3;

static std::ofstream g_log;

void ReportSinr (uint16_t cellId, uint16_t rnti,
                 double sinr, uint16_t bwpId, uint8_t ccId)
{
    double sinrDb = 10.0 * std::log10 (sinr + 1e-9);
    g_log << Simulator::Now ().GetMilliSeconds () << ","
          << sinrDb << ","
          << bwpId  << "\n";
}

int main (int argc, char* argv[])
{
    uint32_t seed        = 1;
    double   simTime     = 10.0;
    double   speed       = 1.5;
    uint32_t gNbNum      = 1;
    uint32_t ueNumPerGnb = 1;

    CommandLine cmd;
    cmd.AddValue ("seed",    "RNG seed",      seed);
    cmd.AddValue ("simTime", "Duration (s)",  simTime);
    cmd.AddValue ("speed",   "UE speed m/s",  speed);
    cmd.Parse (argc, argv);

    RngSeedManager::SetSeed (seed);
    RngSeedManager::SetRun  (seed);

    std::string fname = "baseline_seed" +
                        std::to_string (seed) + ".csv";
    g_log.open (fname);
    g_log << "time_ms,sinr_db,bwp_id\n";

    // ── scenario helper — same as cttc-nr-demo ────────────
    GridScenarioHelper gridScenario;
    gridScenario.SetRows (1);
    gridScenario.SetColumns (gNbNum);
    gridScenario.SetHorizontalBsDistance (5.0);
    gridScenario.SetBsHeight (10.0);
    gridScenario.SetUtHeight (1.5);
    gridScenario.SetSectorization (
        GridScenarioHelper::SINGLE);
    gridScenario.SetBsNumber (gNbNum);
    gridScenario.SetUtNumber (ueNumPerGnb * gNbNum);
    gridScenario.SetScenarioParameters (gridScenario);
    gridScenario.CreateScenario ();

    // Set UE speed
    for (uint32_t i = 0;
         i < gridScenario.GetUserTerminals ().GetN (); i++)
    {
        Ptr<ConstantVelocityMobilityModel> mob =
            gridScenario.GetUserTerminals ().Get (i)
                ->GetObject<ConstantVelocityMobilityModel> ();
        if (mob)
        {
            mob->SetVelocity (Vector (speed, 0.0, 0.0));
        }
    }

    NodeContainer gnbNodes =
        gridScenario.GetBaseStations ();
    NodeContainer ueNodes  =
        gridScenario.GetUserTerminals ();

    // ── NR helpers ────────────────────────────────────────
    Ptr<NrPointToPointEpcHelper> epcHelper =
        CreateObject<NrPointToPointEpcHelper> ();
    Ptr<IdealBeamformingHelper> beamHelper =
        CreateObject<IdealBeamformingHelper> ();
    Ptr<NrHelper> nrHelper =
        CreateObject<NrHelper> ();

    nrHelper->SetBeamformingHelper (beamHelper);
    nrHelper->SetEpcHelper (epcHelper);
    beamHelper->SetBeamformingMethod (
        DirectPathBeamforming::GetTypeId ());

    // ── band — single band, 28 GHz, 100 MHz ──────────────
    BandwidthPartInfoPtrVector allBwps;
    CcBwpCreator ccBwpCreator;
    const uint8_t numCcPerBand = 1;

    CcBwpCreator::SimpleOperationBandConf bandConf (
        28e9, 100e6, numCcPerBand,
        BandwidthPartInfo::UMi_StreetCanyon);

    OperationBandInfo band =
        ccBwpCreator.CreateOperationBandContiguousCc (bandConf);

    nrHelper->InitializeOperationBand (&band);

    // GetAllBwps AFTER InitializeOperationBand
    allBwps = CcBwpCreator::GetAllBwps ({band});

    // ── antenna & PHY ─────────────────────────────────────
    nrHelper->SetGnbAntennaAttribute (
        "NumRows",    UintegerValue (4));
    nrHelper->SetGnbAntennaAttribute (
        "NumColumns", UintegerValue (4));
    nrHelper->SetUeAntennaAttribute (
        "NumRows",    UintegerValue (2));
    nrHelper->SetUeAntennaAttribute (
        "NumColumns", UintegerValue (2));
    nrHelper->SetGnbPhyAttribute (
        "TxPower", DoubleValue (43.0));

    // ── install ───────────────────────────────────────────
    NetDeviceContainer gnbDev =
        nrHelper->InstallGnbDevice (gnbNodes, allBwps);
    NetDeviceContainer ueDev  =
        nrHelper->InstallUeDevice  (ueNodes,  allBwps);

    // Update devices config
    for (auto it = gnbDev.Begin ();
         it != gnbDev.End (); ++it)
    {
        DynamicCast<NrGnbNetDevice> (*it)->UpdateConfig ();
    }
    for (auto it = ueDev.Begin ();
         it != ueDev.End (); ++it)
    {
        DynamicCast<NrUeNetDevice> (*it)->UpdateConfig ();
    }

    // ── internet ──────────────────────────────────────────
    InternetStackHelper internet;
    internet.Install (ueNodes);
    epcHelper->AssignUeIpv4Address (ueDev);
    nrHelper->AttachToClosestEnb (ueDev, gnbDev);

    // ── remote host ───────────────────────────────────────
    NodeContainer remote;
    remote.Create (1);
    internet.Install (remote);

    PointToPointHelper p2p;
    p2p.SetDeviceAttribute  ("DataRate",
                              StringValue ("10Gbps"));
    p2p.SetChannelAttribute ("Delay",
                              StringValue ("1ms"));
    NetDeviceContainer backDev =
        p2p.Install (remote.Get (0),
                     epcHelper->GetPgwNode ());

    Ipv4AddressHelper addr;
    addr.SetBase ("1.0.0.0", "255.0.0.0");
    addr.Assign (backDev);

    Ipv4StaticRoutingHelper routing;
    routing.GetStaticRouting (
        remote.Get (0)->GetObject<Ipv4> ())
        ->AddNetworkRouteTo (
            Ipv4Address ("7.0.0.0"),
            Ipv4Mask ("255.0.0.0"), 1);

    // ── UDP downlink ──────────────────────────────────────
    Ipv4Address ueIp =
        ueNodes.Get (0)->GetObject<Ipv4> ()
               ->GetAddress (1, 0).GetLocal ();

    UdpClientHelper dlClient (ueIp, 1234);
    dlClient.SetAttribute ("Interval",
        TimeValue (MilliSeconds (1)));
    dlClient.SetAttribute ("MaxPackets",
        UintegerValue (0xFFFFFFFF));
    dlClient.SetAttribute ("PacketSize",
        UintegerValue (512));

    auto dlApps = dlClient.Install (remote.Get (0));
    dlApps.Start (Seconds (0.5));
    dlApps.Stop  (Seconds (simTime));

    UdpServerHelper dlSrv (1234);
    dlSrv.Install (ueNodes.Get (0)).Start (Seconds (0.0));

    // ── SINR trace ────────────────────────────────────────
    Config::ConnectWithoutContext (
        "/NodeList/*/DeviceList/*"
        "/$ns3::NrUeNetDevice"
        "/ComponentCarrierMapUe/*/NrUePhy"
        "/DlDataSinr",
        MakeCallback (&ReportSinr));

    Simulator::Stop   (Seconds (simTime));
    Simulator::Run    ();
    Simulator::Destroy ();

    g_log.close ();
    std::cout << "Done. Output: " << fname << std::endl;
    return 0;
}
