package dk.dtu.mobileai;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Bundle;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.telephony.CellInfo;
import android.telephony.PhoneStateListener;
import android.telephony.SignalStrength;
import android.telephony.TelephonyManager;
import android.util.Log;
import android.view.View;
import android.widget.ScrollView;
import android.widget.TextView;
import android.view.View.OnFocusChangeListener;

import java.lang.reflect.Method;
import java.util.List;

import static dk.dtu.mobileai.R.styleable.View;

public class MainActivity extends AppCompatActivity {

    private ScrollView scroller;
    private TextView console;

    private SignalStrength signalStrength;
    private TelephonyManager telephonyManager;
    private PhoneStateListener mListener;
    private final static String LTE_TAG = "LTE_Tag";
    private final static String LTE_SIGNAL_STRENGTH = "getLteSignalStrength";

    private final static int MY_PERMISSIONS_REQUEST = 0;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        telephonyManager = (TelephonyManager) getSystemService(Context.TELEPHONY_SERVICE);
        scroller = (ScrollView) findViewById(R.id.scroller);
        console = (TextView) findViewById(R.id.console);

        // Listener for the signal strength.
        // https://www.programcreek.com/java-api-examples/index.php?api=android.telephony.CellInfo
        mListener = new PhoneStateListener()
        {
            @Override
            public void onSignalStrengthsChanged(SignalStrength sStrength)
            {
                signalStrength = sStrength;
                getLTECqi();
                getLTEsignalStrength();
                Log.i(LTE_TAG, telephonyManager.getAllCellInfo().toString());
            }

            @Override
            public void onCellInfoChanged(List<CellInfo> cellInfoList){
                /*
                if (cellInfoList != null){
                    Log.i(LTE_TAG, "NOT NULL = " + cellInfoList.toString());
                } else {
                    Log.i(LTE_TAG, telephonyManager.getAllCellInfo().toString());
                }
                */
            }
        };


        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_COARSE_LOCATION) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{ Manifest.permission.ACCESS_COARSE_LOCATION }, MY_PERMISSIONS_REQUEST);
        } else {
            Log.i(LTE_TAG, "Permission granted");
            // Register the listener for the telephony manager
            telephonyManager.listen(mListener, PhoneStateListener.LISTEN_SIGNAL_STRENGTHS | PhoneStateListener.LISTEN_CELL_INFO | PhoneStateListener.LISTEN_DATA_CONNECTION_STATE);

        }
        // Example of a call to a native method
        // TextView tv = (TextView) findViewById(R.id.sample_text);
        // tv.setText(stringFromJNI());
    }

    @Override
    public void onRequestPermissionsResult(int requestCode,
                                           String permissions[], int[] grantResults) {
        switch (requestCode) {
            case MY_PERMISSIONS_REQUEST: {
                // If request is cancelled, the result arrays are empty.
                if (grantResults.length > 0
                        && grantResults[0] == PackageManager.PERMISSION_GRANTED) {

                    Log.i(LTE_TAG, "Permission granted");
                    // Register the listener for the telephony manager
                    telephonyManager.listen(mListener, PhoneStateListener.LISTEN_SIGNAL_STRENGTHS | PhoneStateListener.LISTEN_CELL_INFO | PhoneStateListener.LISTEN_DATA_CONNECTION_STATE);

                } else {

                    // permission denied, boo! Disable the
                    // functionality that depends on this permission.
                }
                return;
            }

            // other 'case' lines to check for other
            // permissions this app might request
        }
    }

    private void getLTEsignalStrength() {
        try
        {
            Method[] methods = android.telephony.SignalStrength.class.getMethods();

            for (Method mthd : methods)
            {
                if (mthd.getName().equals(LTE_SIGNAL_STRENGTH))
                {
                    int LTEsignalStrength = (Integer) mthd.invoke(signalStrength, new Object[] {});
                    Log.i(LTE_TAG, "signalStrength = " + LTEsignalStrength);
                    this.print("signalStrength = " + LTEsignalStrength);
                    return;
                }
            }
        }
        catch (Exception e)
        {
            Log.e(LTE_TAG, "Exception: " + e.toString());
        }
    }

    private void getLTECqi() {
        try
        {
            /*
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

            parts[13] = gsm|lte|cdma

            parts[14] = _not really sure what this number is_
             */
            String ssignal = signalStrength.toString();
            String[] parts = ssignal.split(" ");
            String lteCqi = parts[12];

            Log.i(LTE_TAG, "LteCqi = " + lteCqi);
            Log.i(LTE_TAG, "ALL == " + ssignal);
            print("LteCqi = " + lteCqi);
            //print("ALL == " + ssignal);

        }
        catch (Exception e)
        {
            Log.e(LTE_TAG, "Exception: " + e.toString());
        }
    }

    private void print(String text){
        console.append(text);
        console.append("\n");
        scroller.fullScroll(scroller.FOCUS_DOWN);
    }

    /**
     * A native method that is implemented by the 'native-lib' native library,
     * which is packaged with this application.
     */
    public native String stringFromJNI();

    // Used to load the 'native-lib' library on application startup.
    static {
        System.loadLibrary("native-lib");
    }
}
