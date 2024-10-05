package nw_64;

import inet.networklayer.configurator.ipv4.Ipv4NetworkConfigurator;
import inet.networks.base.TsnNetworkBase;
import inet.node.contract.IEthernetNetworkNode;
import inet.node.ethernet.EthernetLink;
import inet.node.inet.Router;
import inet.linklayer.configurator.gatescheduling.contract.IGateScheduleConfigurator;
import inet.visualizer.contract.IIntegratedVisualizer;
import inet.networklayer.configurator.contract.INetworkConfigurator;
import inet.common.packet.recorder.PcapRecorder;


network Nw_64
{

    parameters:
        //*.eth[*].bitrate = default(100Mbps);
        //*.ipv4.arp.typename = default("GlobalArp");
        *.hasUdp = true;

        @display("bgb=650, 650");

    submodules:
        visualizer: <default(firstAvailableOrEmpty("IntegratedCanvasVisualizer"))> like IIntegratedVisualizer if typename != "" {
            @display("p=100,200");
        }
        gateScheduleConfigurator: <default("")> like IGateScheduleConfigurator if typename != "" {
            @display("p=100,125");
        }
        configurator: Ipv4NetworkConfigurator {
            @display("p=100,50");
        }

        MX_3: Router {
            parameters:
                @display("p=1, 1");
            gates:
                ethg[1];
        }

        MX_4: Router {
            parameters:
                @display("p=1, 2");
            gates:
                ethg[1];
        }

        MX_6: Router {
            parameters:
                @display("p=1, 3");
            gates:
                ethg[3];
        }

        MX_7: Router {
            parameters:
                @display("p=1, 4");
            gates:
                ethg[1];
        }

        MX_8: Router {
            parameters:
                @display("p=1, 5");
            gates:
                ethg[2];
        }

        MX_12: Router {
            parameters:
                @display("p=2, 1");
            gates:
                ethg[2];
        }


    connections:
        MX_3.ethg[0] <--> EthernetLink { datarate = 1Gbps; } <--> MX_6.ethg[0];
        MX_4.ethg[0] <--> EthernetLink { datarate = 1Gbps; } <--> MX_6.ethg[1];
        MX_6.ethg[2] <--> EthernetLink { datarate = 1Gbps; } <--> MX_8.ethg[0];
        MX_7.ethg[0] <--> EthernetLink { datarate = 1Gbps; } <--> MX_12.ethg[0];
        MX_8.ethg[1] <--> EthernetLink { datarate = 1Gbps; } <--> MX_12.ethg[1];

}

