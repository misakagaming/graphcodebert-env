package com.j2cg.train;

import com.j2cg.base.CTCoolGenCore;
import com.j2cg.base.CTDBEntityQuery;
import com.j2cg.base.CTDBException;
import com.j2cg.base.CTDBQuery;

public class DYYY0211_TRAIN012 extends CTCoolGenCore {
	protected reference_iyy1_server_data m_InputEntity001;
	protected child m_InputEntity002;

	@Override
	public void initialize(Object... inputs) {
		setDBConnection(inputs);
	}

	@Override
	public void execute(Object... outputs) {
		
		dont_change_return_codes m_LocalEntity001 = new dont_change_return_codes();
		dont_change_reason_codes m_LocalEntity002 = new dont_change_reason_codes();

		child m_Output01 = (child) outputs[0];
		error_iyy1_component m_Output02 = (error_iyy1_component) outputs[1];
		
		try {
				CTDBEntityQuery query = new CTDBEntityQuery(m_DBConnection, parent.class.getName());
				
				child m_Record = new child();
				m_Record.cinstance_id = m_InputEntity002.cinstance_id;
				m_Record.creference_id = m_InputEntity001.reference_id;
				m_Record.ccreate_user_id = m_InputEntity001.userid;
				m_Record.cupdate_user_id = m_InputEntity001.userid;
				m_Record.cparent_pkey_attr_text = m_InputEntity002.cparent_pkey_attr_text;
				m_Record.ckey_attr_num = m_InputEntity002.ckey_attr_num;
				m_Record.csearch_attr_text = m_InputEntity002.csearch_attr_text;
				m_Record.cother_attr_text = m_InputEntity002.cother_attr_text;
				
				query.insert(m_Record);
				switch (query.getErrorCode())
				{
					case CTDBQuery.DBERROR_SUCCESSFUL:
						moveMember(m_Record, m_Output01);
						break;
						
					case CTDBQuery.DBERROR_ALREADYEXISTS:
					    m_Output02.return_code = m_LocalEntity001.n40_obj_create_failed;
						m_Output02.reason_code = m_LocalEntity002.m_122_child_already_exist;
						break;
						
					case CTDBQuery.DBERROR_PERMITTED_VALUE_VIOLATION:
					    m_Output02.return_code = m_LocalEntity001.n40_obj_create_failed;
						m_Output02.reason_code = m_LocalEntity002.m_123_child_attr_value_invalid;
						break;
				}
				
		} catch (CTDBException de) {
			
		}
	}

}
