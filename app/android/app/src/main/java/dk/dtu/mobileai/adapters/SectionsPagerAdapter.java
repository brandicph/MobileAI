package dk.dtu.mobileai.adapters;

import android.provider.ContactsContract;
import android.support.v4.app.Fragment;
import android.support.v4.app.FragmentManager;
import android.support.v4.app.FragmentPagerAdapter;
import android.telephony.CellInfo;
import android.telephony.CellInfoGsm;
import android.telephony.CellInfoLte;
import android.util.Log;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Timer;
import java.util.TimerTask;

import dk.dtu.mobileai.data.DataStore;
import dk.dtu.mobileai.fragments.InfoFragment;
import dk.dtu.mobileai.fragments.MapsFragment;
import dk.dtu.mobileai.fragments.NetworkFragment;
import dk.dtu.mobileai.fragments.PlaceholderFragment;
import dk.dtu.mobileai.listeners.OnCellularModuleChangedListener;
import dk.dtu.mobileai.models.NetworkItem;
import dk.dtu.mobileai.modules.CellularModule;

/**
 * A {@link FragmentPagerAdapter} that returns a fragment corresponding to
 * one of the sections/tabs/pages.
 */
public class SectionsPagerAdapter extends FragmentPagerAdapter {

    private static final String TAG = "SectionsPagerAdapter";

    private Timer networkTimer;
    private int networkInterval = 1000;

    public SectionsPagerAdapter(FragmentManager fm) {
        super(fm);
    }

    // Returns the fragment to display for that page
    @Override
    public Fragment getItem(int position) {
        switch (position) {
            case 0: // Fragment # 0 - This will show FirstFragment
                InfoFragment infoFragment = InfoFragment.newInstance();

                ArrayList<NetworkItem> items = new ArrayList<NetworkItem>();

                DataStore dataStore = DataStore.getInstance();
                CellularModule cellularModule = dataStore.getCellularModule();

                if (cellularModule != null){
                    String deviceName = cellularModule.getDeviceName();
                    String imei = cellularModule.getIMEI();
                    String imsi = cellularModule.getIMSI();

                    items.add(new NetworkItem("Name", deviceName, ""));
                    items.add(new NetworkItem("IMEI", imei, ""));
                    items.add(new NetworkItem("IMSI", imsi, ""));

                    infoFragment.setData(items);
                }

                return infoFragment;
            case 1: // Fragment # 0 - This will show FirstFragment
                final NetworkFragment fragment = NetworkFragment.newInstance();

                CellularModule cm = DataStore.getInstance().getCellularModule();
                if (cm != null){
                    cm.setOnCellularModuleChanged(new OnCellularModuleChangedListener() {
                        @Override
                        public void OnCellularModuleChanged(CellularModule module) {
                            ArrayList<NetworkItem> items = new ArrayList<NetworkItem>();

                            List<CellInfo> cellInfoList = module.getCellInfoList();
                            for (int index = 0; index < cellInfoList.size(); index++) {
                                CellInfo cellInfo = cellInfoList.get(index);

                                if (cellInfo instanceof CellInfoLte) {
                                    // cast to CellInfoLte and call all the CellInfoLte methods you need
                                    // gets RSRP cell signal strength:
                                    int cellSig = ((CellInfoLte) cellInfo).getCellSignalStrength().getDbm();

                                    // Gets the LTE cell identity: (returns 28-bit Cell Identity, Integer.MAX_VALUE if unknown)
                                    int cellID = ((CellInfoLte) cellInfo).getCellIdentity().getCi();

                                    // Gets the LTE MCC: (returns 3-digit Mobile Country Code, 0..999, Integer.MAX_VALUE if unknown)
                                    int cellMcc = ((CellInfoLte) cellInfo).getCellIdentity().getMcc();

                                    // Gets theLTE MNC: (returns 2 or 3-digit Mobile Network Code, 0..999, Integer.MAX_VALUE if unknown)
                                    int cellMnc = ((CellInfoLte) cellInfo).getCellIdentity().getMnc();

                                    // Gets the LTE PCI: (returns Physical Cell Id 0..503, Integer.MAX_VALUE if unknown)
                                    int cellPci = ((CellInfoLte) cellInfo).getCellIdentity().getPci();

                                    // Gets the LTE TAC: (returns 16-bit Tracking Area Code, Integer.MAX_VALUE if unknown)
                                    int cellTac = ((CellInfoLte) cellInfo).getCellIdentity().getTac();

                                    // https://github.com/CellularPrivacy/Android-IMSI-Catcher-Detector
                                    int cellRsrp = ((CellInfoLte) cellInfo).getCellSignalStrength().getRsrp();
                                    int cellRsrq = ((CellInfoLte) cellInfo).getCellSignalStrength().getRsrq();
                                    int cellLevel = ((CellInfoLte) cellInfo).getCellSignalStrength().getLevel();
                                    int cellAsu = ((CellInfoLte) cellInfo).getCellSignalStrength().getAsuLevel();
                                    int cellRssnr = ((CellInfoLte) cellInfo).getCellSignalStrength().getRssnr();
                                    int cellCqi = ((CellInfoLte) cellInfo).getCellSignalStrength().getCqi();

                                    items.add(new NetworkItem("Signal[" + index + "]", String.format("%d", cellSig), "dBm"));
                                    items.add(new NetworkItem("Level[" + index + "]", String.format("%d", cellLevel), "dBm"));
                                    items.add(new NetworkItem("Asu[" + index + "]", String.format("%d", cellAsu), "dBm"));
                                    items.add(new NetworkItem("RSRP[" + index + "]", String.format("%d", cellRsrp), "dBm"));
                                    items.add(new NetworkItem("RSRQ[" + index + "]", String.format("%d", cellRsrq), "dB"));
                                    items.add(new NetworkItem("RSSNR[" + index + "]", String.format("%d", cellRssnr), "dB"));
                                    items.add(new NetworkItem("CQI[" + index + "]", String.format("%d", cellCqi), "Quality"));
                                    items.add(new NetworkItem("ID[" + index + "]", String.format("%d", cellID), "Cell ID"));
                                    items.add(new NetworkItem("MCC[" + index + "]", String.format("%d", cellMcc), "Country Code"));
                                    items.add(new NetworkItem("MNC[" + index + "]", String.format("%d", cellMnc), "Network Code"));
                                    items.add(new NetworkItem("PCI[" + index + "]", String.format("%d", cellPci), "Physical ID"));
                                    items.add(new NetworkItem("TAC[" + index + "]", String.format("%d", cellTac), "Area Code"));

                                    fragment.setData(items);

                                }
                            }
                        }
                    });
                }


                return fragment;
            case 2: // Fragment # 0 - This will show FirstFragment different title
                return MapsFragment.newInstance();
            default:
                return null;
        }
    }

    @Override
    public int getCount() {
        // Show 3 total pages.
        return 3;
    }
}