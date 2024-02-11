package com.j2cg.train;

import com.j2cg.base.CTCoolGenCore;
import com.j2cg.base.CTDBEntityQuery;
import com.j2cg.base.CTDBException;
import com.j2cg.base.CTDBQuery;

public class DYYY0131_TRAIN020 extends CTCoolGenCore {
	protected reference_iyy1_server_data m_InputEntity001;
	protected parent m_InputEntity002;

	protected group_context m_LocalEntity001;
	protected isc1_authorization_info m_LocalEntity002;
	protected g_context_dont_change_text m_LocalEntity003;

	@Override
	public void initialize(Object... inputs) {
		setDBConnection(inputs);
	}

	@Override
	public void execute(Object... outputs) {
		
		dont_change_return_codes m_LocalEntity001 = new dont_change_return_codes();
		dont_change_reason_codes m_LocalEntity002 = new dont_change_reason_codes();

		parent m_Output01 = (parent) outputs[0];
		error_iyy1_component m_Output02 = (error_iyy1_component) outputs[1];
		
		try {
				CTDBEntityQuery query = new CTDBEntityQuery(m_DBConnection, parent.class.getName());
				
				parent m_Record = new parent();
				m_Record.preference_id = m_InputEntity001.reference_id;
				m_Record.pupdate_user_id = m_InputEntity001.userid;
				m_Record.psearch_attr_text = m_InputEntity002.psearch_attr_text;
				m_Record.pother_attr_text = m_InputEntity002.pother_attr_text;
				m_Record.psearch_attr_text = m_InputEntity002.psearch_attr_text;
				m_Record.ptype_tkey_attr_text = m_InputEntity002.ptype_tkey_attr_text;
				
				query.update(m_Record);
				switch (query.getErrorCode())
				{
					case CTDBQuery.DBERROR_SUCCESSFUL:
						moveMember(m_Record, m_Output01);
						break;
						
					case CTDBQuery.DBERROR_NOTUNIQUE:
					    m_Output02.return_code = m_LocalEntity001.n41_obj_update_failed;
						m_Output02.reason_code = m_LocalEntity002.m_102_parent_already_exist;
						break;
						
					case CTDBQuery.DBERROR_PERMITTED_VALUE_VIOLATION:
					    m_Output02.return_code = m_LocalEntity001.n41_obj_update_failed;
						m_Output02.reason_code = m_LocalEntity002.m_103_parent_attr_value_invalid;
						break;
				}
				
		} catch (CTDBException de) {
			
		}

		if (m_LocalEntity002.authority_status_l.equals("N")) {
			
			m_LocalEntity001.setSubscript(3);
			m_LocalEntity003.text_150 = CTCoolGenGlobal.CLIENT_USER_ID;
		}
	}

}
