package dk.dtu.mobileai;

import android.annotation.SuppressLint;
import android.content.Intent;
import android.os.SystemClock;
import android.support.v7.app.ActionBar;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.os.Handler;
import android.view.MotionEvent;
import android.view.View;

import java.util.concurrent.TimeUnit;

public class SplashActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        //SystemClock.sleep(TimeUnit.SECONDS.toMillis(3));

        Intent intent = new Intent(this, TabbedActivity.class);
        startActivity(intent);
        finish();
    }
}
