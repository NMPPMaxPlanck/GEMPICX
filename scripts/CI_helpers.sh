# function for starting a collapsible section
start_details () {
  details_id="${1}"
  details_title="${2:-$details_id}"

  echo "section_start:`date +%s`:${details_id}[collapsed=true]"$'\r\e[0K'"${details_title}"
}

# Function for ending the collapsible section
end_details () {
  details_id="${1}"

  echo "section_end:`date +%s`:${details_id}"$'\r\e[0K'
}
