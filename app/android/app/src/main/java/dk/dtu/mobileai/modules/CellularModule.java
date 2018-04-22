package dk.dtu.mobileai.modules;

import android.content.Context;
import android.telephony.CellInfo;
import android.telephony.CellInfoLte;
import android.telephony.PhoneStateListener;
import android.telephony.TelephonyManager;
import android.util.Log;

import com.jjoe64.graphview.series.DataPoint;

import java.text.DecimalFormat;
import java.util.List;

import dk.dtu.mobileai.data.DataStore;
import dk.dtu.mobileai.data.IApiEndpoint;
import dk.dtu.mobileai.listeners.IOnCellularModuleChangedListener;
import dk.dtu.mobileai.listeners.OnCellularModuleChangedListener;
import dk.dtu.mobileai.models.Location;
import dk.dtu.mobileai.models.Measurement;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

import static android.content.Context.TELEPHONY_SERVICE;


/*
TS 27.007 8.5
Defined values
<rsrp>:
0 -113 dBm or less
1 -111 dBm
2...30 -109... -53 dBm
31 -51 dBm or greater
99 not known or not detectable
 */

/*
The parts[] array will then contain these elements:

part[0] = "Signalstrength:"  _ignore this, it's just the title_
parts[1] = GsmSignalStrength
parts[2] = GsmBitErrorRate
parts[3] = CdmaDbm
parts[4] = CdmaEcio
parts[5] = EvdoDbm
parts[6] = EvdoEcio
parts[7] = EvdoSnr
parts[8] = LteSignalStrength
parts[9] = LteRsrp
parts[10] = LteRsrq
parts[11] = LteRssnr
parts[12] = LteCqi
parts[13] = gsm|lte
parts[14] = _not reall sure what this number is_
 */

public class CellularModule extends PhoneStateListener implements IOnCellularModuleChangedListener {

    private static final String TAG = "CellularModule";

    private static OnCellularModuleChangedListener mListener;

    private Context mContext;
    private TelephonyManager mTelephonyManager;

    private List<CellInfo> cellInfoList;
    int cellSig, cellID, cellMcc, cellMnc, cellPci, cellTac = 0;
    String ueIMSI, ueIMEI, ueDeviceName = null;


    public CellularModule(Context context){
        this.mContext = context;

        ((TelephonyManager) this.mContext.getSystemService(TELEPHONY_SERVICE)).listen(this, this.LISTEN_SIGNAL_STRENGTHS);
        this.mTelephonyManager = (TelephonyManager) mContext.getSystemService(Context.TELEPHONY_SERVICE);
    }

    @Override
    public void onSignalStrengthsChanged(android.telephony.SignalStrength signalStrength) {

        //((TelephonyManager) this.mContext.getSystemService(TELEPHONY_SERVICE)).listen(this, this.LISTEN_SIGNAL_STRENGTHS);
        //this.mTelephonyManager = (TelephonyManager) mContext.getSystemService(Context.TELEPHONY_SERVICE);

        String ltestr = signalStrength.toString();

        Log.d(TAG, ltestr);

        String[] parts = ltestr.split(" ");
        String cellSig2 = parts[9];

        try {
            cellInfoList = mTelephonyManager.getAllCellInfo();
            for (int index = 0; index < cellInfoList.size(); index++) {
                CellInfo cellInfo = cellInfoList.get(index);

                if (cellInfo instanceof CellInfoLte) {
                    // cast to CellInfoLte and call all the CellInfoLte methods you need
                    // gets RSRP cell signal strength:
                    cellSig = ((CellInfoLte) cellInfo).getCellSignalStrength().getDbm();

                    // Gets the LTE cell identity: (returns 28-bit Cell Identity, Integer.MAX_VALUE if unknown)
                    cellID = ((CellInfoLte) cellInfo).getCellIdentity().getCi();

                    // Gets the LTE MCC: (returns 3-digit Mobile Country Code, 0..999, Integer.MAX_VALUE if unknown)
                    cellMcc = ((CellInfoLte) cellInfo).getCellIdentity().getMcc();

                    // Gets theLTE MNC: (returns 2 or 3-digit Mobile Network Code, 0..999, Integer.MAX_VALUE if unknown)
                    cellMnc = ((CellInfoLte) cellInfo).getCellIdentity().getMnc();

                    // Gets the LTE PCI: (returns Physical Cell Id 0..503, Integer.MAX_VALUE if unknown)
                    cellPci = ((CellInfoLte) cellInfo).getCellIdentity().getPci();

                    // Gets the LTE TAC: (returns 16-bit Tracking Area Code, Integer.MAX_VALUE if unknown)
                    cellTac = ((CellInfoLte) cellInfo).getCellIdentity().getTac();

                    // Gets the IMEI
                    ueIMEI = mTelephonyManager.getImei();

                    // Gets the Device Name
                    ueDeviceName = getDeviceName();

                    // Gets the IMSI
                    ueIMSI = mTelephonyManager.getSubscriberId();

                    DataStore dataStore = DataStore.getInstance();

                    int cellRsrp = ((CellInfoLte) cellInfo).getCellSignalStrength().getRsrp();
                    int cellRsrq = ((CellInfoLte) cellInfo).getCellSignalStrength().getRsrq();
                    int cellLevel = ((CellInfoLte) cellInfo).getCellSignalStrength().getLevel();
                    int cellAsu = ((CellInfoLte) cellInfo).getCellSignalStrength().getAsuLevel();
                    int cellRssnr = ((CellInfoLte) cellInfo).getCellSignalStrength().getRssnr();
                    int cellCqi = ((CellInfoLte) cellInfo).getCellSignalStrength().getCqi();

                    if (dataStore.getApiSync()){
                        IApiEndpoint apiService = dataStore.retrofit.create(IApiEndpoint.class);

                        Measurement measurement = new Measurement();
                        measurement.setEntity(dataStore.getApiEndpoint() + "entities/" + dataStore.getApiEntityId() + "/");

                        measurement.setAntenna(index);
                        measurement.setAsu(cellAsu);
                        measurement.setCellId(cellID);
                        measurement.setLevel(cellLevel);
                        measurement.setMcc(cellMcc);
                        measurement.setMnc(cellMnc);
                        measurement.setCqi(cellCqi);
                        measurement.setPci(cellPci);
                        measurement.setRsrp(cellRsrp);
                        measurement.setRsrq(cellRsrq);
                        measurement.setRssnr(cellRssnr);
                        measurement.setSignal(cellSig);
                        measurement.setTac(cellTac);

                        Call<Measurement> call = apiService.createMeasurement(DataStore.getInstance().getApiEntityId(), measurement);

                        try {
                            call.enqueue(new Callback<Measurement>() {
                                @Override
                                public void onResponse(Call<Measurement> call, Response<Measurement> response) {
                                    Log.d(TAG, response.toString());
                                }

                                @Override
                                public void onFailure(Call<Measurement> call, Throwable t) {
                                    Log.d(TAG, t.toString());
                                }
                            });
                        } catch (Exception e ){
                            Log.e(TAG, e.getMessage());
                        }
                    }

                }
            }
            if (mListener != null){
                mListener.OnCellularModuleChanged(this);
            }
            DataStore.getInstance().notifyDataChanged();
        } catch (SecurityException e){
            //RequestMultiplePermission();
        } catch (Exception e) {
            Log.e(TAG, e.getMessage());
        }

        super.onSignalStrengthsChanged(signalStrength);

    }

    public List<CellInfo> getCellInfoList(){
        return cellInfoList;
    }

    public String getDeviceName() {
        String manufacturer = android.os.Build.MANUFACTURER;
        String model = android.os.Build.MODEL;
        if (model.startsWith(manufacturer)) {
            return capitalize(model);
        } else {
            return capitalize(manufacturer) + " " + model;
        }
    }

    public String getIMEI() {
        return ueIMEI;
    }

    public String getIMSI() {
        return ueIMSI;
    }

    private String capitalize(String s) {
        if (s == null || s.length() == 0) {
            return "";
        }
        char first = s.charAt(0);
        if (Character.isUpperCase(first)) {
            return s;
        } else {
            return Character.toUpperCase(first) + s.substring(1);
        }
    }

    @Override
    public void setOnCellularModuleChanged(OnCellularModuleChangedListener listener) {
        mListener = listener;
    }
}
