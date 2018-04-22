package dk.dtu.mobileai.adapters;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ArrayAdapter;
import android.widget.TextView;

import java.util.ArrayList;

import dk.dtu.mobileai.R;
import dk.dtu.mobileai.models.NetworkItem;

public class NetworkAdapter extends ArrayAdapter<NetworkItem> {
    public NetworkAdapter(Context context, ArrayList<NetworkItem> networkItems) {
        super(context, 0, networkItems);
    }

    @Override
    public View getView(int position, View convertView, ViewGroup parent) {
        // Get the data item for this position
        NetworkItem networkItem = getItem(position);
        // Check if an existing view is being reused, otherwise inflate the view
        if (convertView == null) {
            convertView = LayoutInflater.from(getContext()).inflate(R.layout.network_item, parent, false);
        }
        // Lookup view for data population
        TextView tvName = (TextView) convertView.findViewById(R.id.tvName);
        TextView tvValue = (TextView) convertView.findViewById(R.id.tvValue);
        TextView tvUnit = (TextView) convertView.findViewById(R.id.tvUnit);
        // Populate the data into the template view using the data object
        tvName.setText(networkItem.name);
        tvValue.setText(networkItem.value);
        tvUnit.setText(networkItem.unit);
        // Return the completed view to render on screen
        return convertView;
    }
}
